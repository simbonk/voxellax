// WebGL2 voxel video player with off-axis ("window") perspective.
// The phone screen is treated as a window into the scene: the eye position
// (from head tracking, the auto path, or dragging) moves the asymmetric
// frustum so the voxels shift with real parallax.
import { identity, multiply, translation, scaling, frustum } from './math.js';
import { DEPTH_SCALE } from './format.js';

const CM_PER_CSS_PX = 0.0175; // rough physical size of a CSS pixel on phones/tablets

const VS = `#version 300 es
precision highp float;
in vec2 aUV;
in float aDepth;
in vec3 aColor;
uniform mat4 uXform;
uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uTan;
uniform vec3 uCenter;
uniform float uDepthK;
uniform float uCut;
uniform float uCell;
uniform float uScale;
uniform float uPix;
uniform float uSizeMul;
out vec3 vColor;
out float vShade;
void main() {
  if (aDepth < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  float z = aDepth / ${DEPTH_SCALE.toFixed(1)};
  vec3 p = vec3((aUV.x * 2.0 - 1.0) * uTan.x * z, (1.0 - aUV.y * 2.0) * uTan.y * z, -z);
  vec3 w = (uXform * vec4(p, 1.0)).xyz;
  vec3 d = w - uCenter;
  if (d.z < -uCut) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  w = uCenter + d * vec3(1.0, 1.0, uDepthK);
  vec4 v = uView * uModel * vec4(w, 1.0);
  gl_Position = uProj * v;
  float worldSize = z * uCell * uScale * uSizeMul;
  gl_PointSize = clamp(worldSize * uPix / max(-v.z, 0.01), 1.0, 96.0);
  vColor = aColor;
  vShade = clamp(1.0 + d.z / (uCut * 1.5), 0.45, 1.0);
}`;

const FS = `#version 300 es
precision mediump float;
in vec3 vColor;
in float vShade;
out vec4 outColor;
void main() {
  vec2 pc = abs(gl_PointCoord - 0.5);
  float edge = max(pc.x, pc.y);
  float bevel = edge > 0.40 ? 0.78 : 1.0;
  float light = 1.0 + 0.10 * (0.5 - gl_PointCoord.y);
  outColor = vec4(vColor * vShade * bevel * light, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

export class VoxelPlayer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.loc = {};
    for (const n of ['aUV', 'aDepth', 'aColor']) this.loc[n] = gl.getAttribLocation(prog, n);
    for (const n of ['uXform', 'uModel', 'uView', 'uProj', 'uTan', 'uCenter', 'uDepthK', 'uCut', 'uCell', 'uScale', 'uPix', 'uSizeMul']) {
      this.loc[n] = gl.getUniformLocation(prog, n);
    }
    this.settings = {
      mode: 'auto', // 'head' | 'auto' | 'drag'
      views: 'both', // 'both' | '0' | '1'
      gain: 1.6,
      depthK: 1.0,
      pointSize: 1.15,
      cut: 60,
      zoom: 1.0,
    };
    this.clip = null;
    this.gpuViews = [];
    this.playing = false;
    this.time = 0;
    this.lastFrame = -1;
    this.headTracker = null;
    this.drag = { x: 0, y: 0 };
    this.eye = { x: 0, y: 0, z: 38 };
    this.onFrame = null;
    this.bindPointer();
    this.raf = requestAnimationFrame((t) => this.tick(t));
  }

  bindPointer() {
    const c = this.canvas;
    let down = null;
    c.addEventListener('pointerdown', (e) => {
      down = { x: e.clientX, y: e.clientY, ox: this.drag.x, oy: this.drag.y };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', (e) => {
      if (!down) return;
      const k = CM_PER_CSS_PX * 3;
      this.drag.x = down.ox - (e.clientX - down.x) * k;
      this.drag.y = down.oy + (e.clientY - down.y) * k;
      if (this.settings.mode !== 'drag') this.dragOverride = performance.now();
    });
    const up = () => (down = null);
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
  }

  load(clip) {
    const gl = this.gl;
    this.stop();
    for (const v of this.gpuViews) {
      gl.deleteVertexArray(v.vao);
      [v.uvBuf, v.depthBuf, v.colorBuf].forEach((b) => gl.deleteBuffer(b));
    }
    this.clip = clip;
    this.duration = clip.frameCount / clip.fps;
    this.gpuViews = clip.views.map((view) => {
      const N = view.gw * view.gh;
      const uv = new Float32Array(N * 2);
      for (let y = 0; y < view.gh; y++) {
        for (let x = 0; x < view.gw; x++) {
          const i = y * view.gw + x;
          uv[i * 2] = (x + 0.5) / view.gw;
          uv[i * 2 + 1] = (y + 0.5) / view.gh;
        }
      }
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const uvBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(this.loc.aUV);
      gl.vertexAttribPointer(this.loc.aUV, 2, gl.FLOAT, false, 0, 0);
      const depthBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, depthBuf);
      gl.bufferData(gl.ARRAY_BUFFER, N * 2, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.loc.aDepth);
      gl.vertexAttribPointer(this.loc.aDepth, 1, gl.UNSIGNED_SHORT, false, 0, 0);
      const colorBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, colorBuf);
      gl.bufferData(gl.ARRAY_BUFFER, N * 3, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(this.loc.aColor);
      gl.vertexAttribPointer(this.loc.aColor, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      gl.bindVertexArray(null);
      return { view, N, vao, uvBuf, depthBuf, colorBuf, frame: -1 };
    });
    if (clip.views.length === 2) {
      this.settings.views = clip.mergeOk ? 'both' : String(clip.views[1].faceFrames > clip.views[0].faceFrames ? 1 : 0);
    } else {
      this.settings.views = '0';
    }
    this.time = 0;
    this.lastFrame = -1;
  }

  // ---- audio + clock ------------------------------------------------------

  ensureAudio() {
    if (!this.clip?.audio || this.audioBuffer) return;
    try {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.actx = this.actx || new Ctx();
      const { rate, pcm } = this.clip.audio;
      const outRate = this.actx.sampleRate;
      const n = Math.floor((pcm.length * outRate) / rate);
      const buf = this.actx.createBuffer(1, n, outRate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < n; i++) {
        const x = (i * rate) / outRate;
        const i0 = Math.floor(x), fr = x - i0;
        const a = pcm[i0] || 0, b = pcm[Math.min(pcm.length - 1, i0 + 1)] || 0;
        ch[i] = (a + (b - a) * fr) / 32768;
      }
      this.audioBuffer = buf;
    } catch (e) {
      console.warn('audio unavailable', e);
    }
  }

  play() {
    if (!this.clip) return;
    this.ensureAudio();
    if (this.audioBuffer && this.actx) {
      this.actx.resume?.();
      const src = this.actx.createBufferSource();
      src.buffer = this.audioBuffer;
      src.loop = true;
      src.loopEnd = Math.min(this.audioBuffer.duration, this.duration);
      src.connect(this.actx.destination);
      const offset = this.time % this.duration;
      src.start(0, Math.min(offset, this.audioBuffer.duration - 0.01));
      this.audioSrc = src;
      this.clockBase = this.actx.currentTime - offset;
    } else {
      this.clockBase = performance.now() / 1000 - this.time;
    }
    this.playing = true;
  }

  stop() {
    if (this.audioSrc) {
      try { this.audioSrc.stop(); } catch {}
      this.audioSrc = null;
    }
    this.playing = false;
  }

  pause() {
    this.stop();
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
    return this.playing;
  }

  currentTime() {
    if (!this.playing) return this.time;
    const now = this.audioSrc ? this.actx.currentTime : performance.now() / 1000;
    return (now - this.clockBase) % this.duration;
  }

  // ---- head tracking --------------------------------------------------------

  async setMode(mode) {
    this.settings.mode = mode;
    if (mode === 'head') {
      if (!this.headTracker) {
        const { HeadTracker } = await import('./headtrack.js');
        this.headTracker = new HeadTracker();
        try {
          await this.headTracker.start();
        } catch (e) {
          this.headTracker = null;
          this.settings.mode = 'auto';
          throw e;
        }
      }
    } else if (this.headTracker) {
      this.headTracker.stop();
      this.headTracker = null;
    }
  }

  recenter() {
    this.headTracker?.recenter();
    this.drag = { x: 0, y: 0 };
  }

  // ---- render ---------------------------------------------------------------

  updateEye(tSec, W, H) {
    const s = this.settings;
    let target;
    if (s.mode === 'head' && this.headTracker?.pos) {
      const p = this.headTracker.pos;
      target = { x: p.x * s.gain, y: p.y * s.gain, z: Math.min(80, Math.max(20, p.z)) };
    } else if (s.mode === 'drag' || (this.dragOverride && performance.now() - this.dragOverride < 2500)) {
      target = { x: this.drag.x, y: this.drag.y, z: 38 };
    } else if (s.mode === 'head') {
      target = { x: 0, y: 0, z: 38 }; // waiting for a face
    } else {
      // Auto: slow figure-of-eight around the window.
      target = {
        x: Math.sin(tSec * 0.55) * W * 0.55,
        y: Math.sin(tSec * 0.37 + 0.7) * H * 0.3,
        z: 38 + Math.sin(tSec * 0.23) * 5,
      };
      this.drag = { x: target.x, y: target.y };
    }
    const k = 0.2;
    this.eye.x += (target.x - this.eye.x) * k;
    this.eye.y += (target.y - this.eye.y) * k;
    this.eye.z += (target.z - this.eye.z) * k;
  }

  tick(ts) {
    this.raf = requestAnimationFrame((t) => this.tick(t));
    const gl = this.gl;
    const c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(1, Math.round(c.clientWidth * dpr));
    const ch = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== cw || c.height !== ch) {
      c.width = cw;
      c.height = ch;
    }
    gl.viewport(0, 0, cw, ch);
    gl.clearColor(0.03, 0.03, 0.05, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.clip) return;

    const t = this.currentTime();
    if (!this.playing) this.time = t;
    else this.time = t;
    const frame = Math.min(this.clip.frameCount - 1, Math.floor(t * this.clip.fps));

    const W = c.clientWidth * CM_PER_CSS_PX;
    const H = c.clientHeight * CM_PER_CSS_PX;
    this.updateEye(ts / 1000, W, H);
    const e = this.eye;
    const near = 1, far = 3000;
    const l = ((-W / 2 - e.x) * near) / e.z;
    const r = ((W / 2 - e.x) * near) / e.z;
    const b = ((-H / 2 - e.y) * near) / e.z;
    const tp = ((H / 2 - e.y) * near) / e.z;
    const proj = frustum(l, r, b, tp, near, far);
    const view = translation(-e.x, -e.y, -e.z);
    const uPix = (ch * near) / (tp - b);

    const s = this.settings;
    const minWH = Math.min(W, H);
    const scale = (s.zoom * 0.8 * minWH) / 20;
    const showBoth = s.views === 'both' && this.gpuViews.length === 2;
    const baseCenter = showBoth ? this.clip.views[0].center : null;

    gl.useProgram(this.prog);
    gl.enable(gl.DEPTH_TEST);
    gl.uniformMatrix4fv(this.loc.uProj, false, proj);
    gl.uniformMatrix4fv(this.loc.uView, false, view);
    gl.uniform1f(this.loc.uPix, uPix);
    gl.uniform1f(this.loc.uScale, scale);
    gl.uniform1f(this.loc.uSizeMul, s.pointSize);
    gl.uniform1f(this.loc.uDepthK, s.depthK);
    gl.uniform1f(this.loc.uCut, s.cut);

    const id = identity();
    const drawList = showBoth
      ? [0, 1]
      : [this.gpuViews.length === 1 || s.views === 'both' ? 0 : Number(s.views)];
    drawList.forEach((vi) => {
      const gv = this.gpuViews[vi];
      const v = gv.view;
      if (gv.frame !== frame) {
        gl.bindBuffer(gl.ARRAY_BUFFER, gv.depthBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, v.depth.subarray(frame * gv.N, (frame + 1) * gv.N));
        gl.bindBuffer(gl.ARRAY_BUFFER, gv.colorBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, v.color.subarray(frame * gv.N * 3, (frame + 1) * gv.N * 3));
        gv.frame = frame;
      }
      const center = baseCenter || v.center;
      const xform = showBoth ? v.xform.subarray(frame * 16, frame * 16 + 16) : id;
      // Scene: put the face centre just behind the screen.
      const model = multiply(
        translation(0, 0, -0.15 * minWH),
        multiply(scaling(scale), translation(-center[0], -center[1], -center[2])),
      );
      gl.uniformMatrix4fv(this.loc.uXform, false, xform);
      gl.uniformMatrix4fv(this.loc.uModel, false, model);
      gl.uniform2f(this.loc.uTan, v.tanX, v.tanY);
      gl.uniform3f(this.loc.uCenter, center[0], center[1], center[2]);
      gl.uniform1f(this.loc.uCell, (2 * v.tanY) / v.gh);
      gl.bindVertexArray(gv.vao);
      gl.drawArrays(gl.POINTS, 0, gv.N);
    });
    gl.bindVertexArray(null);
    if (this.onFrame && frame !== this.lastFrame) this.onFrame(frame, t);
    this.lastFrame = frame;
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.stop();
    this.headTracker?.stop();
  }
}

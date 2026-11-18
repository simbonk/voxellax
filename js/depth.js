// Monocular depth with Depth Anything V2 (small) running in the browser through
// Transformers.js / ONNX Runtime Web. WebGPU when available, WASM otherwise.
// We do our own pre-processing so the input resolution can be kept small enough
// for a phone.
import { TRANSFORMERS_URL, DEPTH_MODEL_ID, MOCK_DEPTH } from './libs.js';

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

export class DepthEstimator {
  constructor() {
    this.model = null;
    this.T = null;
    this.device = 'none';
    this.dtype = '';
  }

  async load(onProgress = () => {}) {
    if (MOCK_DEPTH) {
      this.device = 'mock';
      return;
    }
    this.T = await import(TRANSFORMERS_URL);
    const { AutoModelForDepthEstimation, env } = this.T;
    env.allowLocalModels = false;
    env.useBrowserCache = true;

    const hasWebGPU = !!(navigator.gpu && (await navigator.gpu.requestAdapter().catch(() => null)));
    const attempts = hasWebGPU
      ? [['webgpu', 'fp16'], ['webgpu', 'fp32'], ['wasm', 'q8'], ['wasm', 'fp32']]
      : [['wasm', 'q8'], ['wasm', 'fp32']];

    let lastErr = null;
    for (const [device, dtype] of attempts) {
      try {
        onProgress({ status: 'loading', device, dtype });
        const files = {};
        const model = await AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID, {
          device,
          dtype,
          progress_callback: (p) => {
            if (p.status === 'progress' && p.total) {
              files[p.file] = [p.loaded, p.total];
              let l = 0, t = 0;
              for (const [a, b] of Object.values(files)) { l += a; t += b; }
              onProgress({ status: 'download', device, dtype, loaded: l, total: t });
            }
          },
        });
        this.model = model;
        this.device = device;
        this.dtype = dtype;
        // Warm-up run catches unsupported-operator / dtype problems early.
        await this.run(new ImageData(new Uint8ClampedArray(28 * 28 * 4).fill(128), 28, 28));
        return;
      } catch (e) {
        console.warn(`Depth model failed on ${device}/${dtype}`, e);
        lastErr = e;
        try { await this.model?.dispose?.(); } catch {}
        this.model = null;
      }
    }
    throw lastErr || new Error('Could not load the depth model');
  }

  /**
   * imageData: RGBA ImageData whose width/height are multiples of 14.
   * Returns { data: Float32Array (relative inverse depth, bigger = closer), width, height }.
   */
  async run(imageData) {
    const { width: w, height: h, data } = imageData;
    if (this.device === 'mock') return mockDepth(imageData);
    const n = w * h;
    const px = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      px[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
      px[n + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
      px[2 * n + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
    }
    const input = new this.T.Tensor('float32', px, [1, 3, h, w]);
    const out = await this.model({ pixel_values: input });
    const t = out.predicted_depth ?? Object.values(out)[0];
    const dims = t.dims;
    const oh = dims[dims.length - 2], ow = dims[dims.length - 1];
    const d = t.data instanceof Float32Array ? t.data : Float32Array.from(t.data);
    const result = { data: new Float32Array(d), width: ow, height: oh };
    t.dispose?.();
    return result;
  }
}

// Fake depth for testing without the network model: a soft blob in the centre
// plus a brightness term, so the pipeline has something plausible to chew on.
function mockDepth({ width: w, height: h, data }) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dx = (x / w - 0.5) / 0.3, dy = (y / h - 0.45) / 0.3;
      const blob = Math.exp(-(dx * dx + dy * dy));
      const i = y * w + x;
      const lum = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 765;
      out[i] = 1 + 4 * blob + 0.5 * lum;
    }
  }
  return { data: out, width: w, height: h };
}

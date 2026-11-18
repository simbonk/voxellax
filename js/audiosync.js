// Audio alignment of two recordings of the same moment.
// Decodes both soundtracks with the Web Audio API and finds the time offset with
// GCC-PHAT (phase-transform weighted cross-correlation), which is robust to the
// two microphones sounding different.

export const SYNC_RATE = 16000; // decode rate; also the rate of exported audio
const CORR_RATE = 4000; // rate used for the correlation itself

export async function decodeAudio(file) {
  const buf = await file.arrayBuffer();
  let ctx;
  try {
    ctx = new OfflineAudioContext(1, 1, SYNC_RATE);
  } catch {
    ctx = new OfflineAudioContext(1, 1, 44100); // older Safari: fixed rates only
  }
  const audio = await new Promise((resolve, reject) => {
    const p = ctx.decodeAudioData(buf, resolve, reject);
    if (p && p.then) p.then(resolve, reject);
  });
  const n = audio.length;
  const mono = new Float32Array(n);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const ch = audio.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += ch[i] / audio.numberOfChannels;
  }
  return resample(mono, audio.sampleRate, SYNC_RATE);
}

// Box-filtered linear resampler (good enough for sync and 16 kHz speech).
export function resample(x, from, to) {
  if (from === to) return x;
  const ratio = from / to;
  const n = Math.floor(x.length / ratio);
  const out = new Float32Array(n);
  const half = Math.max(0, Math.floor(ratio / 2));
  for (let i = 0; i < n; i++) {
    const c = Math.floor(i * ratio);
    let s = 0, k = 0;
    for (let j = c - half; j <= c + half; j++) {
      if (j >= 0 && j < x.length) { s += x[j]; k++; }
    }
    out[i] = k ? s / k : 0;
  }
  return out;
}

// In-place iterative radix-2 complex FFT.
function fft(re, im, inverse) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (2 * Math.PI / len) * (inverse ? 1 : -1);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const halfLen = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < halfLen; j++) {
        const a = i + j, b = a + halfLen;
        const xr = re[b] * cr - im[b] * ci;
        const xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/**
 * Returns { offset, confidence } where offset = tA - tB in seconds: the moment
 * heard at time tB in clip B is heard at tB + offset in clip A.
 * maxLag bounds the search (seconds).
 */
export function findOffset(a16, b16, maxLag = 30) {
  const maxSecs = 60; // only correlate the first minute of each clip
  const a = resample(a16.subarray(0, maxSecs * SYNC_RATE), SYNC_RATE, CORR_RATE);
  const b = resample(b16.subarray(0, maxSecs * SYNC_RATE), SYNC_RATE, CORR_RATE);
  let n = 1;
  while (n < a.length + b.length) n <<= 1;
  const ar = new Float64Array(n), ai = new Float64Array(n);
  const br = new Float64Array(n), bi = new Float64Array(n);
  // Remove DC and apply a gentle pre-emphasis (speech onsets matter most).
  const prep = (src, dst) => {
    let mean = 0;
    for (let i = 0; i < src.length; i++) mean += src[i];
    mean /= src.length || 1;
    let prev = 0;
    for (let i = 0; i < src.length; i++) {
      const v = src[i] - mean;
      dst[i] = v - 0.9 * prev;
      prev = v;
    }
  };
  prep(a, ar);
  prep(b, br);
  fft(ar, ai, false);
  fft(br, bi, false);
  // Cross-power spectrum A * conj(B), PHAT weighted.
  for (let i = 0; i < n; i++) {
    const r = ar[i] * br[i] + ai[i] * bi[i];
    const im = ai[i] * br[i] - ar[i] * bi[i];
    const mag = Math.hypot(r, im) + 1e-9;
    ar[i] = r / mag;
    ai[i] = im / mag;
  }
  fft(ar, ai, true);
  const lagMax = Math.min(n / 2 - 1, Math.round(maxLag * CORR_RATE));
  let best = -Infinity, bestK = 0;
  const val = (k) => ar[(k + n) % n];
  for (let k = -lagMax; k <= lagMax; k++) {
    const v = val(k);
    if (v > best) { best = v; bestK = k; }
  }
  // Second-best peak at least 100 ms away -> confidence ratio.
  let second = 1e-9, sum = 0, cnt = 0;
  const guard = Math.round(0.1 * CORR_RATE);
  for (let k = -lagMax; k <= lagMax; k++) {
    const v = val(k);
    sum += Math.abs(v); cnt++;
    if (Math.abs(k - bestK) > guard && v > second) second = v;
  }
  // Parabolic interpolation for sub-sample precision.
  const y0 = val(bestK - 1), y1 = val(bestK), y2 = val(bestK + 1);
  const den = y0 - 2 * y1 + y2;
  const frac = den !== 0 ? 0.5 * (y0 - y2) / den : 0;
  return {
    offset: (bestK + frac) / CORR_RATE,
    confidence: best / second,
    peakToMean: best / (sum / cnt),
  };
}

// Small column-major matrix / quaternion helpers (OpenGL conventions).

export function identity() {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}

export function translation(x, y, z) {
  const m = identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

export function scaling(x, y = x, z = x) {
  const m = identity();
  m[0] = x; m[5] = y; m[10] = z;
  return m;
}

export function frustum(l, r, b, t, n, f) {
  const m = new Float32Array(16);
  m[0] = (2 * n) / (r - l);
  m[5] = (2 * n) / (t - b);
  m[8] = (r + l) / (r - l);
  m[9] = (t + b) / (t - b);
  m[10] = -(f + n) / (f - n);
  m[11] = -1;
  m[14] = (-2 * f * n) / (f - n);
  return m;
}

export function quatToMat3(q) {
  const [w, x, y, z] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
    2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x),
    2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y),
  ]; // column-major 3x3
}

// Similarity transform (scale s, rotation q, translation t) -> mat4.
export function similarityToMat4(s, q, t) {
  const r = quatToMat3(q);
  const m = new Float32Array(16);
  m[0] = s * r[0]; m[1] = s * r[1]; m[2] = s * r[2];
  m[4] = s * r[3]; m[5] = s * r[4]; m[6] = s * r[5];
  m[8] = s * r[6]; m[9] = s * r[7]; m[10] = s * r[8];
  m[12] = t[0]; m[13] = t[1]; m[14] = t[2]; m[15] = 1;
  return m;
}

export function applySim(s, q, t, p) {
  const r = quatToMat3(q);
  return [
    s * (r[0] * p[0] + r[3] * p[1] + r[6] * p[2]) + t[0],
    s * (r[1] * p[0] + r[4] * p[1] + r[7] * p[2]) + t[1],
    s * (r[2] * p[0] + r[5] * p[1] + r[8] * p[2]) + t[2],
  ];
}

// Jacobi eigen-decomposition of a symmetric 4x4 (array of rows).
function jacobiEigen4(A) {
  const a = A.map((r) => r.slice());
  const v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] * a[p][q];
    if (off < 1e-18) break;
    for (let p = 0; p < 4; p++) {
      for (let q = p + 1; q < 4; q++) {
        if (Math.abs(a[p][q]) < 1e-30) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 4; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

/**
 * Horn's closed-form similarity alignment: finds s, q, t minimising
 * |dst - (s R src + t)|. src/dst: arrays of [x,y,z]. Optional weights.
 */
export function alignSimilarity(src, dst, weights) {
  const n = src.length;
  let W = 0;
  const cs = [0, 0, 0], cd = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    W += w;
    for (let k = 0; k < 3; k++) { cs[k] += w * src[i][k]; cd[k] += w * dst[i][k]; }
  }
  for (let k = 0; k < 3; k++) { cs[k] /= W; cd[k] /= W; }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let ss = 0, sd = 0;
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1;
    const a = [src[i][0] - cs[0], src[i][1] - cs[1], src[i][2] - cs[2]];
    const b = [dst[i][0] - cd[0], dst[i][1] - cd[1], dst[i][2] - cd[2]];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) S[r][c] += w * a[r] * b[c];
    ss += w * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    sd += w * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  }
  const [[Sxx, Sxy, Sxz], [Syx, Syy, Syz], [Szx, Szy, Szz]] = S;
  const N = [
    [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
    [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
    [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
    [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
  ];
  let q = jacobiEigen4(N);
  const qn = Math.hypot(...q) || 1;
  q = q.map((x) => x / qn);
  if (q[0] < 0) q = q.map((x) => -x);
  const s = ss > 0 ? Math.sqrt(sd / ss) : 1;
  const rc = applySim(s, q, [0, 0, 0], cs);
  const t = [cd[0] - rc[0], cd[1] - rc[1], cd[2] - rc[2]];
  return { s, q, t };
}

export function median(arr) {
  if (!arr.length) return NaN;
  const a = Array.from(arr).sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

export function percentile(arr, p) {
  if (!arr.length) return NaN;
  const a = Array.from(arr).sort((x, y) => x - y);
  return a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))];
}

// The .voxl file: a small binary container, gzip-compressed when the browser
// supports CompressionStream.
//
//   "VOXL" | u32 version | u32 headerBytes | header JSON (utf-8, padded to 4)
//   followed by raw chunks; header.chunks[name] = { offset, length } (bytes,
//   relative to the start of the chunk area, every chunk 4-byte aligned).
//
// Per view: depth (Uint16, cm * DEPTH_SCALE, 0 = empty), color (RGB Uint8),
// xform (Float32 4x4 per frame, column-major; maps this camera's space into the
// left camera's space). Optional audio: Int16 mono PCM.

export const DEPTH_SCALE = 50;
const MAGIC = [0x56, 0x4f, 0x58, 0x4c];
const VERSION = 1;

export async function encodeVoxl(clip) {
  const chunks = [];
  const header = {
    app: 'voxellax',
    version: VERSION,
    created: new Date().toISOString(),
    fps: clip.fps,
    frameCount: clip.frameCount,
    mergeOk: clip.mergeOk,
    mergeInfo: clip.mergeInfo,
    views: [],
    audio: null,
    chunks: {},
  };
  let offset = 0;
  const add = (name, typed) => {
    const bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
    header.chunks[name] = { offset, length: bytes.byteLength };
    chunks.push(bytes);
    const pad = (4 - (bytes.byteLength % 4)) % 4;
    if (pad) chunks.push(new Uint8Array(pad));
    offset += bytes.byteLength + pad;
  };
  clip.views.forEach((v, i) => {
    header.views.push({
      name: v.name,
      gw: v.gw,
      gh: v.gh,
      tanX: v.tanX,
      tanY: v.tanY,
      center: v.center,
      faceFrames: v.faceFrames,
    });
    add(`v${i}.depth`, v.depth);
    add(`v${i}.color`, v.color);
    add(`v${i}.xform`, v.xform);
  });
  if (clip.audio) {
    header.audio = { rate: clip.audio.rate, length: clip.audio.pcm.length };
    add('audio', clip.audio.pcm);
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const jpad = (4 - (json.byteLength % 4)) % 4;
  const pre = new Uint8Array(12);
  pre.set(MAGIC, 0);
  const dv = new DataView(pre.buffer);
  dv.setUint32(4, VERSION, true);
  dv.setUint32(8, json.byteLength + jpad, true);
  const parts = [pre, json, new Uint8Array(jpad), ...chunks];
  let blob = new Blob(parts, { type: 'application/octet-stream' });
  if (typeof CompressionStream !== 'undefined') {
    const stream = blob.stream().pipeThrough(new CompressionStream('gzip'));
    blob = await new Response(stream).blob();
    blob = new Blob([blob], { type: 'application/octet-stream' });
  }
  return blob;
}

export async function decodeVoxl(fileOrBlob) {
  let buf = await fileOrBlob.arrayBuffer();
  let u8 = new Uint8Array(buf);
  if (u8[0] === 0x1f && u8[1] === 0x8b) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('This browser cannot decompress .voxl files (needs iOS 16.4+ or a recent browser).');
    }
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    buf = await new Response(stream).arrayBuffer();
    u8 = new Uint8Array(buf);
  }
  if (!MAGIC.every((b, i) => u8[i] === b)) throw new Error('Not a Voxellax (.voxl) file.');
  const dv = new DataView(buf);
  const version = dv.getUint32(4, true);
  if (version > VERSION) throw new Error(`File is version ${version}; this app reads up to ${VERSION}.`);
  const hlen = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + hlen)).replace(/\0+$/, ''));
  const base = 12 + hlen;
  const get = (name, Type) => {
    const c = header.chunks[name];
    if (!c) return null;
    return new Type(buf.slice(base + c.offset, base + c.offset + c.length));
  };
  const clip = {
    fps: header.fps,
    frameCount: header.frameCount,
    mergeOk: header.mergeOk,
    mergeInfo: header.mergeInfo,
    views: header.views.map((v, i) => ({
      ...v,
      depth: get(`v${i}.depth`, Uint16Array),
      color: get(`v${i}.color`, Uint8Array),
      xform: get(`v${i}.xform`, Float32Array),
    })),
    audio: header.audio ? { rate: header.audio.rate, pcm: get('audio', Int16Array) } : null,
  };
  return clip;
}

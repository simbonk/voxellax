// Pulls the AAC soundtrack out of an MP4/MOV file in JavaScript.
//
// Why: recent iPhones record a second "spatial audio" track (APAC) next to the
// normal AAC one, and some browsers refuse to decodeAudioData() the whole .mov
// because of it (or because the file is large). Demuxing just the AAC track and
// handing the decoder a plain ADTS .aac stream sidesteps both problems.

const te = new TextDecoder('latin1');

function boxes(dv, start, end) {
  const out = [];
  let p = start;
  while (p + 8 <= end) {
    let size = dv.getUint32(p);
    const type = te.decode(new Uint8Array(dv.buffer, dv.byteOffset + p + 4, 4));
    let header = 8;
    if (size === 1) {
      size = Number(dv.getBigUint64(p + 8));
      header = 16;
    } else if (size === 0) {
      size = end - p;
    }
    if (size < header || p + size > end) break;
    out.push({ type, start: p, dataStart: p + header, end: p + size });
    p += size;
  }
  return out;
}

const child = (dv, box, type) => boxes(dv, box.dataStart, box.end).find((b) => b.type === type);

function path(dv, box, types) {
  let b = box;
  for (const t of types) {
    b = b && child(dv, b, t);
    if (!b) return null;
  }
  return b;
}

// Find a box by type anywhere below [start, end) (used for esds inside wave).
function findDeep(dv, start, end, type, depth = 0) {
  if (depth > 6) return null;
  for (const b of boxes(dv, start, end)) {
    if (b.type === type) return b;
    const inner = findDeep(dv, b.dataStart, b.end, type, depth + 1);
    if (inner) return inner;
  }
  return null;
}

function readDescLen(u8, p) {
  let len = 0, n = 0;
  while (n < 4) {
    const b = u8[p + n];
    len = (len << 7) | (b & 0x7f);
    n++;
    if (!(b & 0x80)) break;
  }
  return { len, size: n };
}

// AudioSpecificConfig from an esds box.
function parseEsds(dv, esds) {
  const u8 = new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength);
  let p = esds.dataStart + 4; // version + flags
  const end = esds.end;
  while (p < end) {
    const tag = u8[p];
    const { len, size } = readDescLen(u8, p + 1);
    const body = p + 1 + size;
    if (tag === 0x03) {
      const flags = u8[body + 2];
      let q = body + 3;
      if (flags & 0x80) q += 2;
      if (flags & 0x40) q += 1 + u8[q];
      if (flags & 0x20) q += 2;
      p = q;
      continue;
    }
    if (tag === 0x04) {
      p = body + 13;
      continue;
    }
    if (tag === 0x05) return u8.slice(body, body + len);
    p = body + len;
  }
  return null;
}

const FREQS = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

function parseAsc(asc) {
  let bit = 0;
  const read = (n) => {
    let v = 0;
    for (let i = 0; i < n; i++, bit++) v = (v << 1) | ((asc[bit >> 3] >> (7 - (bit & 7))) & 1);
    return v;
  };
  let objectType = read(5);
  if (objectType === 31) objectType = 32 + read(6);
  let freqIndex = read(4);
  let rate = FREQS[freqIndex];
  if (freqIndex === 15) {
    rate = read(24);
    freqIndex = FREQS.reduce((best, f, i) => (Math.abs(f - rate) < Math.abs(FREQS[best] - rate) ? i : best), 0);
  }
  const channels = read(4);
  return { objectType, freqIndex, rate, channels };
}

/**
 * Returns { adts: Uint8Array, asc, frames: [Uint8Array], sampleRate, channels,
 *           startTrim (s), leadPad (s) } for the first AAC track, or null.
 */
export function extractAac(buffer) {
  const dv = new DataView(buffer);
  const top = boxes(dv, 0, dv.byteLength);
  const moov = top.find((b) => b.type === 'moov');
  if (!moov) return null;
  const mvhd = child(dv, moov, 'mvhd');
  const movieScale = mvhd ? dv.getUint32(mvhd.dataStart + (dv.getUint8(mvhd.dataStart) === 1 ? 20 : 12)) : 600;

  for (const trak of boxes(dv, moov.dataStart, moov.end).filter((b) => b.type === 'trak')) {
    const mdia = child(dv, trak, 'mdia');
    const hdlr = mdia && child(dv, mdia, 'hdlr');
    if (!hdlr) continue;
    const handler = te.decode(new Uint8Array(buffer, hdlr.dataStart + 8, 4));
    if (handler !== 'soun') continue;
    const stbl = path(dv, mdia, ['minf', 'stbl']);
    const stsd = stbl && child(dv, stbl, 'stsd');
    if (!stsd) continue;
    const entry = boxes(dv, stsd.dataStart + 8, stsd.end)[0];
    if (!entry || entry.type !== 'mp4a') continue;
    const version = dv.getUint16(entry.dataStart + 8);
    const extra = version === 1 ? 16 : version === 2 ? 36 : 0;
    const esds = findDeep(dv, entry.dataStart + 28 + extra, entry.end, 'esds');
    if (!esds) continue;
    const asc = parseEsds(dv, esds);
    if (!asc || asc.length < 2) continue;
    const cfg = parseAsc(asc);
    if (!cfg.rate || !cfg.channels) continue;

    // Sample sizes.
    const stsz = child(dv, stbl, 'stsz');
    const fixed = dv.getUint32(stsz.dataStart + 4);
    const count = dv.getUint32(stsz.dataStart + 8);
    const sizes = new Uint32Array(count);
    for (let i = 0; i < count; i++) sizes[i] = fixed || dv.getUint32(stsz.dataStart + 12 + i * 4);
    // Chunk offsets.
    const stco = child(dv, stbl, 'stco');
    const co64 = child(dv, stbl, 'co64');
    const chunkOffsets = [];
    if (stco) {
      const n = dv.getUint32(stco.dataStart + 4);
      for (let i = 0; i < n; i++) chunkOffsets.push(dv.getUint32(stco.dataStart + 8 + i * 4));
    } else if (co64) {
      const n = dv.getUint32(co64.dataStart + 4);
      for (let i = 0; i < n; i++) chunkOffsets.push(Number(dv.getBigUint64(co64.dataStart + 8 + i * 8)));
    } else continue;
    // Samples per chunk.
    const stsc = child(dv, stbl, 'stsc');
    const nsc = dv.getUint32(stsc.dataStart + 4);
    const runs = [];
    for (let i = 0; i < nsc; i++) {
      const o = stsc.dataStart + 8 + i * 12;
      runs.push({ first: dv.getUint32(o), per: dv.getUint32(o + 4) });
    }
    const frames = [];
    let s = 0;
    for (let c = 0; c < chunkOffsets.length && s < count; c++) {
      let per = runs[0].per;
      for (const r of runs) if (r.first <= c + 1) per = r.per;
      let off = chunkOffsets[c];
      for (let k = 0; k < per && s < count; k++, s++) {
        if (off + sizes[s] > buffer.byteLength) break;
        frames.push(new Uint8Array(buffer, off, sizes[s]));
        off += sizes[s];
      }
    }
    if (!frames.length) continue;

    // Edit list: leading empty edit (delay) and media start (priming trim).
    const mdhd = child(dv, mdia, 'mdhd');
    const mdhdV1 = dv.getUint8(mdhd.dataStart) === 1;
    const mediaScale = dv.getUint32(mdhd.dataStart + (mdhdV1 ? 20 : 12));
    let leadPad = 0, startTrim = 0;
    const elst = path(dv, trak, ['edts', 'elst']);
    if (elst) {
      const v1 = dv.getUint8(elst.dataStart) === 1;
      const n = dv.getUint32(elst.dataStart + 4);
      let o = elst.dataStart + 8;
      for (let i = 0; i < n; i++) {
        const dur = v1 ? Number(dv.getBigUint64(o)) : dv.getUint32(o);
        const mt = v1 ? Number(dv.getBigInt64(o + 8)) : dv.getInt32(o + 4);
        o += v1 ? 20 : 12;
        if (mt === -1) leadPad += dur / movieScale;
        else {
          startTrim = mt / mediaScale;
          break;
        }
      }
    }

    // Wrap as ADTS so decodeAudioData accepts it as a plain .aac stream.
    const profile = Math.min(3, Math.max(0, (cfg.objectType === 5 || cfg.objectType === 29 ? 2 : cfg.objectType) - 1));
    const chan = Math.min(7, cfg.channels);
    let total = 0;
    for (const f of frames) total += f.length + 7;
    const adts = new Uint8Array(total);
    let p = 0;
    for (const f of frames) {
      const len = f.length + 7;
      adts[p] = 0xff;
      adts[p + 1] = 0xf1;
      adts[p + 2] = (profile << 6) | (cfg.freqIndex << 2) | (chan >> 2);
      adts[p + 3] = ((chan & 3) << 6) | (len >> 11);
      adts[p + 4] = (len >> 3) & 0xff;
      adts[p + 5] = ((len & 7) << 5) | 0x1f;
      adts[p + 6] = 0xfc;
      adts.set(f, p + 7);
      p += len;
    }
    return { adts, asc, frames, sampleRate: cfg.rate, channels: cfg.channels, startTrim, leadPad, mediaScale };
  }
  return null;
}

// Decode raw AAC frames with WebCodecs (used if decodeAudioData rejects ADTS).
export async function decodeAacWebCodecs(info) {
  if (typeof AudioDecoder === 'undefined') throw new Error('WebCodecs AudioDecoder unavailable');
  const config = {
    codec: 'mp4a.40.2',
    sampleRate: info.sampleRate,
    numberOfChannels: info.channels,
    description: info.asc,
  };
  const support = await AudioDecoder.isConfigSupported(config);
  if (!support.supported) throw new Error('AAC not supported by AudioDecoder');
  const chunks = [];
  let total = 0;
  let rate = info.sampleRate;
  let err = null;
  const dec = new AudioDecoder({
    output: (ad) => {
      rate = ad.sampleRate;
      const n = ad.numberOfFrames, ch = ad.numberOfChannels;
      const mono = new Float32Array(n);
      const tmp = new Float32Array(n);
      for (let c = 0; c < ch; c++) {
        ad.copyTo(tmp, { planeIndex: c, format: 'f32-planar' });
        for (let i = 0; i < n; i++) mono[i] += tmp[i] / ch;
      }
      chunks.push(mono);
      total += n;
      ad.close();
    },
    error: (e) => (err = e),
  });
  dec.configure(config);
  const frameDur = (1024 / info.sampleRate) * 1e6;
  info.frames.forEach((f, i) => {
    dec.decode(new EncodedAudioChunk({ type: 'key', timestamp: Math.round(i * frameDur), data: f }));
  });
  await dec.flush();
  dec.close();
  if (err) throw err;
  const out = new Float32Array(total);
  let p = 0;
  for (const c of chunks) {
    out.set(c, p);
    p += c.length;
  }
  return { pcm: out, sampleRate: rate };
}

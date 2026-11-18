import { decodeAudio, findOffset } from './audiosync.js';
import { processClip, seekTo, DETAIL } from './processor.js';
import { encodeVoxl, decodeVoxl } from './format.js';
import { VoxelPlayer } from './player.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

// ---------------------------------------------------------------- navigation
function show(id) {
  $$('.screen').forEach((s) => (s.hidden = s.id !== id));
  window.scrollTo(0, 0);
}
$$('[data-go]').forEach((b) => b.addEventListener('click', () => show(b.dataset.go)));

let toastTimer;
function toast(msg, ms = 3500) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), ms);
}

// ---------------------------------------------------------------- state
const state = {
  files: [null, null],
  videos: [null, null],
  audio: [null, null], // Float32Array @16k or null
  audioErr: [null, null],
  audioDone: [false, false],
  offset: 0, // tA - tB
  syncConfidence: null,
  clip: null,
};

window.voxellax = state; // handy for debugging / tests

$('#btn-new').addEventListener('click', () => show('create'));

// ---------------------------------------------------------------- video slots
function makeVideo(file) {
  const v = document.createElement('video');
  v.muted = true;
  v.playsInline = true;
  v.setAttribute('playsinline', '');
  v.setAttribute('webkit-playsinline', '');
  v.preload = 'auto';
  v.src = URL.createObjectURL(file);
  return v;
}

async function prepareVideo(v) {
  await new Promise((res, rej) => {
    if (v.readyState >= 1) return res();
    v.addEventListener('loadedmetadata', res, { once: true });
    v.addEventListener('error', () => rej(new Error('This browser could not open the video.')), { once: true });
  });
  // iOS only decodes frames after a play() from a user gesture context; a muted
  // play/pause primes it.
  try {
    await v.play();
    v.pause();
  } catch {}
  if (v.readyState < 2) {
    await new Promise((res) => {
      v.addEventListener('loadeddata', res, { once: true });
      setTimeout(res, 3000);
    });
  }
}

$$('.card').forEach((card) => {
  const slot = Number(card.dataset.slot);
  const input = $('input[type=file]', card);
  input.addEventListener('change', async () => {
    const file = input.files[0];
    if (!file) return;
    state.files[slot] = file;
    const box = $('.vidbox', card);
    const meta = $('.meta', card);
    box.innerHTML = '';
    const v = makeVideo(file);
    box.appendChild(v);
    state.videos[slot] = v;
    state.audio[slot] = null;
    state.audioDone[slot] = false;
    meta.textContent = 'Opening…';
    try {
      await prepareVideo(v);
      meta.textContent = `${file.name} · ${v.duration.toFixed(1)} s · ${v.videoWidth}×${v.videoHeight}`;
    } catch (e) {
      meta.textContent = e.message;
      return;
    }
    meta.textContent += ' · reading sound…';
    try {
      state.audio[slot] = await decodeAudio(file);
      state.audioErr[slot] = null;
      meta.textContent = meta.textContent.replace(' · reading sound…', '');
    } catch (e) {
      console.warn(e);
      state.audioErr[slot] = e;
      meta.textContent = meta.textContent.replace(' · reading sound…', ' · no readable sound');
    }
    state.audioDone[slot] = true;
    onSlotsChanged();
  });
});

function onSlotsChanged() {
  const [a, b] = state.videos;
  if (!a || !b) return;
  $('#sync-panel').hidden = false;
  if (!state.audioDone[0] || !state.audioDone[1]) {
    $('#sync-status').className = 'status';
    $('#sync-status').textContent = 'Reading the sound…';
    return;
  }
  $('#clip-panel').hidden = false;
  runSync();
}

function runSync() {
  const st = $('#sync-status');
  const [a, b] = state.audio;
  if (!a || !b) {
    state.offset = 0;
    st.className = 'status warn';
    st.textContent = 'Could not read the sound from both videos, so they start lined up at 0 s. Use the buttons to nudge, then preview.';
    updateOffset();
    return;
  }
  st.className = 'status';
  st.textContent = 'Matching the sound…';
  setTimeout(() => {
    const r = findOffset(a, b);
    state.offset = r.offset;
    state.syncConfidence = r.confidence;
    const good = r.confidence > 1.5 && r.peakToMean > 8;
    st.className = good ? 'status ok' : 'status warn';
    const who = r.offset >= 0 ? 'left video started first' : 'right video started first';
    st.textContent = good
      ? `Matched: ${who} by ${Math.abs(r.offset).toFixed(3)} s.`
      : `Weak match (${who} by ${Math.abs(r.offset).toFixed(3)} s?). Preview and nudge if needed.`;
    updateOffset();
  }, 30);
}

$$('[data-nudge]').forEach((b) =>
  b.addEventListener('click', () => {
    state.offset += Number(b.dataset.nudge);
    updateOffset();
  }),
);

// Overlap of the two clips given the offset (tA = tB + offset).
function overlap() {
  const [a, b] = state.videos;
  const tA0 = Math.max(0, state.offset);
  const tB0 = tA0 - state.offset;
  const dur = Math.max(0, Math.min(a.duration - tA0, b.duration - tB0));
  return { tA0, tB0, dur };
}

function updateOffset() {
  $('#offset-label').textContent = `${state.offset >= 0 ? '+' : ''}${state.offset.toFixed(3)} s`;
  const { dur } = overlap();
  const start = $('#start'), len = $('#length');
  start.max = Math.max(0, dur - 1).toFixed(1);
  len.max = Math.max(1, Math.min(20, dur)).toFixed(1);
  if (Number(start.value) > Number(start.max)) start.value = start.max;
  if (Number(len.value) > Number(len.max)) len.value = len.max;
  updateClipLabels();
}

function updateClipLabels() {
  const { dur } = overlap();
  const start = Number($('#start').value);
  let len = Number($('#length').value);
  len = Math.min(len, Math.max(0.5, dur - start));
  $('#start-label').textContent = `${start.toFixed(1)} s of ${dur.toFixed(1)} s overlap`;
  $('#length-label').textContent = `${len.toFixed(1)} s`;
  const fps = Number($('#fps').value);
  const views = $('#views').value === 'both' ? 2 : 1;
  const frames = Math.round(len * fps);
  const perFrame = { low: 0.6, medium: 1.0, high: 1.8 }[$('#detail').value];
  const est = frames * views * perFrame;
  $('#estimate').textContent = `${frames} frames × ${views} camera${views > 1 ? 's' : ''}. Roughly ${Math.max(1, Math.round(est / 60))}–${Math.max(2, Math.round((est * 3) / 60))} min on a phone (much faster with WebGPU).`;
}
['#start', '#length', '#fps', '#detail', '#views'].forEach((s) => $(s).addEventListener('input', updateClipLabels));

// Side-by-side preview of the synced clips.
let previewTimer = null;
$('#btn-preview-sync').addEventListener('click', async () => {
  const pair = $('#preview-pair');
  clearTimeout(previewTimer);
  pair.innerHTML = '';
  const { tA0, tB0 } = overlap();
  const start = Number($('#start').value);
  const pv = state.files.map((f) => makeVideo(f));
  pv.forEach((v) => pair.appendChild(v));
  pv[0].muted = false;
  await Promise.all(pv.map((v) => prepareVideo(v)));
  await Promise.all([seekTo(pv[0], tA0 + start), seekTo(pv[1], tB0 + start)]);
  await Promise.all(pv.map((v) => v.play().catch(() => {})));
  previewTimer = setTimeout(() => pv.forEach((v) => v.pause()), 5000);
});

// ---------------------------------------------------------------- processing
let abortCtl = null;
let wakeLock = null;

$('#btn-process').addEventListener('click', async () => {
  const { tA0, tB0, dur } = overlap();
  const start = Number($('#start').value);
  const length = Math.min(Number($('#length').value), Math.max(0.5, dur - start));
  const fps = Number($('#fps').value);
  const detail = $('#detail').value;
  const sel = $('#views').value;
  const all = [
    { name: 'left', video: state.videos[0], t0: tA0 + start },
    { name: 'right', video: state.videos[1], t0: tB0 + start },
  ];
  const sources = sel === 'both' ? all : [all[Number(sel)]];
  const audioSlot = state.audio[0] ? 0 : state.audio[1] ? 1 : -1;
  const audio = audioSlot >= 0 ? { pcm16k: state.audio[audioSlot], t0: all[audioSlot].t0 } : null;

  show('process');
  const vids = $('#proc-videos');
  vids.innerHTML = '';
  sources.forEach((s) => vids.appendChild(s.video));
  $('#proc-bar').style.width = '0%';
  $('#proc-eta').textContent = '';
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch {}
  abortCtl = new AbortController();
  try {
    const clip = await processClip({
      sources,
      length,
      fps,
      detail,
      audio,
      previewCanvas: $('#proc-preview'),
      signal: abortCtl.signal,
      onProgress: (p) => {
        $('#proc-stage').textContent = p.stage;
        if (p.fraction != null) $('#proc-bar').style.width = `${(p.fraction * 100).toFixed(1)}%`;
        if (p.eta != null) $('#proc-eta').textContent = `About ${formatSecs(p.eta)} left`;
      },
    });
    state.clip = clip;
    openPlayer(clip, true);
  } catch (e) {
    console.error(e);
    if (e.name !== 'AbortError') alertBox(`Conversion failed: ${e.message || e}`);
    show('create');
  } finally {
    // Put the videos back in their cards.
    $$('.card').forEach((card) => {
      const v = state.videos[Number(card.dataset.slot)];
      if (v && v.parentElement !== $('.vidbox', card)) $('.vidbox', card).appendChild(v);
    });
    try { await wakeLock?.release(); } catch {}
    wakeLock = null;
  }
});

$('#btn-cancel').addEventListener('click', () => abortCtl?.abort());

function formatSecs(s) {
  s = Math.round(s);
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}

function alertBox(msg) {
  toast(msg, 8000);
}

// ---------------------------------------------------------------- player
let player = null;

function getPlayer() {
  if (!player) {
    player = new VoxelPlayer($('#gl'));
  }
  return player;
}

function openPlayer(clip, fresh) {
  show('player');
  const p = getPlayer();
  p.load(clip);
  state.clip = clip;
  $('#views-seg').hidden = clip.views.length < 2;
  syncSeg('#views-seg', 'views', p.settings.views);
  syncSeg('#mode-seg', 'mode', p.settings.mode);
  $('#btn-play').textContent = '▶ Play';
  const parts = [`${clip.frameCount} frames at ${clip.fps} fps`, `${clip.views.length} camera${clip.views.length > 1 ? 's' : ''}`];
  if (clip.views.length === 2) {
    const mi = clip.mergeInfo;
    parts.push(
      clip.mergeOk
        ? `merged (${mi?.alignedFrames}/${mi?.frames} frames aligned, ~${mi?.medianResidualCm} cm error)`
        : `merge unreliable (${mi?.alignedFrames ?? 0}/${mi?.frames ?? clip.frameCount} frames aligned), showing one camera`,
    );
  }
  if (!clip.audio) parts.push('no sound');
  $('#clip-info').textContent = parts.join(' · ');
  if (fresh) {
    hudMsg(clip.views.length === 2 && !clip.mergeOk
      ? 'The two cameras could not be lined up reliably, so one is shown. Try “Both” anyway in the bottom bar.'
      : 'Done! Tap Save to keep the .voxl file.', 5000);
  }
}

let hudTimer;
function hudMsg(text, ms = 3000) {
  const m = $('#hud-msg');
  m.textContent = text;
  m.hidden = false;
  clearTimeout(hudTimer);
  if (ms) hudTimer = setTimeout(() => (m.hidden = true), ms);
}

function syncSeg(sel, key, value) {
  $$(`${sel} button`).forEach((b) => b.classList.toggle('on', b.dataset[key] === value));
}

$('#btn-play').addEventListener('click', () => {
  const playing = getPlayer().toggle();
  $('#btn-play').textContent = playing ? '❚❚ Pause' : '▶ Play';
});

$$('#mode-seg button').forEach((b) =>
  b.addEventListener('click', async () => {
    const p = getPlayer();
    const mode = b.dataset.mode;
    syncSeg('#mode-seg', 'mode', mode);
    if (mode === 'head') hudMsg('Starting the front camera…', 0);
    try {
      await p.setMode(mode);
      if (mode === 'head') hudMsg('Head tracking on. Move your head to look around. Recenter is in ⚙︎.');
      else if (mode === 'drag') hudMsg('Drag on the screen to move the view.');
      else hudMsg('Auto camera motion.');
    } catch (e) {
      console.warn(e);
      syncSeg('#mode-seg', 'mode', 'auto');
      hudMsg(`Head tracking is unavailable (${e.message || e.name}). Using Auto.`, 5000);
    }
  }),
);

$$('#views-seg button').forEach((b) =>
  b.addEventListener('click', () => {
    getPlayer().settings.views = b.dataset.views;
    syncSeg('#views-seg', 'views', b.dataset.views);
  }),
);

$$('[data-set]').forEach((inp) =>
  inp.addEventListener('input', () => {
    getPlayer().settings[inp.dataset.set] = Number(inp.value);
  }),
);

$('#btn-settings').addEventListener('click', () => ($('#settings').hidden = !$('#settings').hidden));
$('#btn-close-settings').addEventListener('click', () => ($('#settings').hidden = true));
$('#btn-recenter').addEventListener('click', () => {
  getPlayer().recenter();
  hudMsg('Recentered.');
});

$('#btn-home').addEventListener('click', async () => {
  const p = getPlayer();
  p.pause();
  await p.setMode('auto').catch(() => {});
  syncSeg('#mode-seg', 'mode', 'auto');
  $('#settings').hidden = true;
  show('home');
});

$('#btn-save').addEventListener('click', async () => {
  if (!state.clip) return;
  hudMsg('Packing file…', 0);
  try {
    const blob = await encodeVoxl(state.clip);
    const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
    const name = `voxellax-${stamp}.voxl`;
    const file = new File([blob], name, { type: 'application/octet-stream' });
    const mb = (blob.size / 1e6).toFixed(1);
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      try {
        await navigator.share({ files: [file], title: 'Voxellax video' });
        hudMsg(`Shared ${name} (${mb} MB). Choose “Save to Files” to keep it.`);
        return;
      } catch (e) {
        if (e.name === 'AbortError') {
          hudMsg('Save cancelled.');
          return;
        }
      }
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 60000);
    hudMsg(`Saved ${name} (${mb} MB).`);
  } catch (e) {
    console.error(e);
    hudMsg(`Could not save: ${e.message}`, 6000);
  }
});

// ---------------------------------------------------------------- open file
$('#open-file').addEventListener('change', async (ev) => {
  const f = ev.target.files[0];
  ev.target.value = '';
  if (!f) return;
  try {
    toast('Opening…');
    const clip = await decodeVoxl(f);
    openPlayer(clip, false);
    hudMsg('Tap Play. Try Head mode for parallax.');
  } catch (e) {
    console.error(e);
    toast(`Could not open that file: ${e.message}`, 6000);
  }
});

// Expose a few things for the automated test.
window.voxellaxDebug = { encodeVoxl, decodeVoxl, openPlayer, getPlayer, DETAIL };

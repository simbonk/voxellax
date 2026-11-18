// MediaPipe Face Landmarker, used both for processing (finding the face in each
// source frame) and for playback head tracking.
import { MEDIAPIPE_URL, MEDIAPIPE_WASM, FACE_MODEL_URL } from './libs.js';

let filesetPromise = null;
let mpModule = null;

async function fileset() {
  if (!filesetPromise) {
    filesetPromise = (async () => {
      mpModule = await import(MEDIAPIPE_URL);
      return mpModule.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
    })();
  }
  return filesetPromise;
}

export async function createFaceLandmarker(runningMode) {
  const fs = await fileset();
  const opts = {
    baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate: 'GPU' },
    runningMode,
    numFaces: 1,
    minFaceDetectionConfidence: 0.4,
    minFacePresenceConfidence: 0.4,
    minTrackingConfidence: 0.4,
  };
  try {
    return await mpModule.FaceLandmarker.createFromOptions(fs, opts);
  } catch (e) {
    console.warn('Face landmarker GPU delegate failed, using CPU', e);
    opts.baseOptions.delegate = 'CPU';
    return mpModule.FaceLandmarker.createFromOptions(fs, opts);
  }
}

// Iris centres (478-point model) with eye-corner fallback.
export function eyePoints(lm) {
  if (lm.length >= 478) return [lm[468], lm[473]];
  const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
  return [mid(lm[33], lm[133]), mid(lm[362], lm[263])];
}

// A stable subset of landmarks used for the two-view alignment: skip the
// contour (indices that sit on the silhouette have unreliable depth).
export const ALIGN_LANDMARKS = (() => {
  const out = [];
  for (let i = 0; i < 468; i += 3) out.push(i);
  return out;
})();

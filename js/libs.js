// Third-party library locations. Everything runs in the browser; nothing is
// processed on a server. `?libbase=/some/path/` swaps the CDN for local copies
// (used by the headless test harness).
const q = new URLSearchParams(location.search);
const libBase = q.get('libbase');

export const MOCK_DEPTH = q.has('mockdepth');

export const TRANSFORMERS_URL = libBase
  ? `${libBase}transformers.min.js`
  : 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js';

export const MEDIAPIPE_URL = libBase
  ? `${libBase}tasks-vision/vision_bundle.mjs`
  : 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';

export const MEDIAPIPE_WASM = libBase
  ? `${libBase}tasks-vision/wasm`
  : 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';

export const FACE_MODEL_URL = libBase
  ? `${libBase}face_landmarker.task`
  : 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export const DEPTH_MODEL_ID = 'onnx-community/depth-anything-v2-small';

# Voxellax

Turn two handheld selfie videos of the same moment into a chunky 3D voxel video,
then play it back with head-tracked parallax on your phone.

Everything runs in the browser (JavaScript, WebGL, WebAssembly/WebGPU). There is no
server processing and no build step. The site is plain static files.

## How it works

1. **Upload two clips.** For example, a selfie video from an iPad in your left hand
   and one from an iPhone in your right hand, both started before you say hello.
   iPhone `.mov` (HEVC) works in iOS Safari.
2. **Audio sync.** Both soundtracks are decoded with the Web Audio API and lined up
   with GCC-PHAT cross-correlation. You can nudge the offset by a frame and preview
   both clips side by side.
3. **3D conversion** (on the device):
   - Frames are sampled at 6–15 fps from the overlapping part.
   - [Depth Anything V2 Small](https://huggingface.co/onnx-community/depth-anything-v2-small)
     estimates relative depth for every frame via Transformers.js (WebGPU if
     available, otherwise WASM). Inputs are downsampled (252–448 px on the long
     side) to stay within a phone's limits.
   - MediaPipe Face Landmarker finds the face. Its inter-pupil distance gives
     real-world scale, so each view becomes a point cloud in centimetres.
   - **Two-view merge (best effort):** the same face landmarks are seen by both
     cameras, so a similarity transform (Horn's method with outlier rejection,
     smoothed over time) maps the right camera into the left camera's space. If too
     few frames line up, the result falls back to one camera. Both views are still
     saved, so you can switch between Both / Left / Right in the player.
4. **Export.** The result is a `.voxl` file (gzipped binary: per-frame depth +
   colour grids, per-frame alignment transforms and 16 kHz mono audio). On iPhone,
   **Save** opens the share sheet. Choose "Save to Files". Open it later with
   "Open a saved .voxl file".
5. **Playback.** WebGL2 renders each point as a square voxel. The screen acts as a
   window with an off-axis (asymmetric) frustum:
   - **Head:** front camera + MediaPipe face tracking moves the viewpoint with your head.
   - **Auto:** a slow automatic camera path (also the fallback when head tracking is
     unavailable).
   - **Drag:** move the viewpoint with your finger.
   Settings (⚙︎) has parallax strength, depth exaggeration, voxel size, background
   cut, zoom and head recentering.

Expect the result to be noisy and chunky. Monocular depth is a guess per frame, and
two uncalibrated handheld cameras can only be aligned approximately.

## Deploy on Vercel

1. Go to [vercel.com/new](https://vercel.com/new) and import the `simbonk/voxellax` repo.
2. Framework preset: **Other**. Leave the build command and output directory empty
   (the repo root is served as-is). No environment variables are needed.
3. Click **Deploy**. Every push to `main` redeploys automatically, and pull requests
   get preview URLs.

The camera (head tracking) needs HTTPS, which Vercel provides. On iOS, allow
camera access when asked. To reset a denied permission, go to Settings → Safari →
Camera.

## Local development

Serve the folder with any static server, for example `npx serve .`, and open it on
`localhost`. Camera access needs `localhost` or HTTPS. Two test flags help:

- `?mockdepth`: skips the depth model and uses a synthetic depth map.
- `?libbase=/vendor/`: loads Transformers.js, MediaPipe and the face model from a
  local folder instead of the CDNs.

## Files

| File | Purpose |
| --- | --- |
| `index.html`, `css/style.css` | UI (home, create, processing, player) |
| `js/main.js` | Screen flow and controls |
| `js/audiosync.js` | Audio decode + GCC-PHAT offset |
| `js/processor.js` | Frame sampling, depth, face scale, two-view alignment |
| `js/depth.js` | Depth Anything via Transformers.js |
| `js/face.js`, `js/headtrack.js` | MediaPipe face landmarker, head tracking |
| `js/player.js` | WebGL2 voxel renderer with off-axis projection |
| `js/format.js` | `.voxl` read/write |
| `js/math.js` | Matrices, quaternions, Horn alignment |

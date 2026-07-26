# Loop progress note (updated 2026-07-26 — paused by user, resume ~3h later)

PAUSED BY USER. On resume: re-enter the same /loop (prompt below), read this
file, continue from "Next". Immediate next step: clone
asigatchov/fast-volleyball-tracking-inference into vendor/fast-vb-tracking
(last attempt failed: `fatal: fetch-pack: invalid index-pack output` — retry;
if it persists, download the repo ZIP via codeload.github.com instead of git).
Note: the other session pivoted architecture to a FastAPI local web app
(farmclip.py, roadmap "Decisions" revised); pipeline stages now plug into its
run_pipeline() stub — same stage code, different entry point.

/loop prompt to re-enter:
/loop Build the farmclip Python pipeline to north star (pipeline scope only, per docs/plans/roadmap.md Decisions): uv project; auto court calibration via existing keypoint models + solvePnP; TrackNet-vs-YOLO ball bake-off + ballistic 3D lifting; player detect/track -> ground positions; then 3D skeletons; emit per-frame scene JSON; self-verify each stage by rendering overlay debug frames to out/debug/ and inspecting them. Input: first minute of examples/*.mp4. CPU-only. Do not touch index.html (other session owns the viewer). Follow the loop-usage-guard per-iteration check. Loop until pipeline is complete.

# Original note (2026-07-26 ~03:55 ET)

Session 57939641-f863-4dae-a2e7-aeb9100df366 running /loop to build the Python
pipeline (docs/plans/roadmap.md "Decisions"). Usage-guard active; user reported
80% of 5h limit (usage.json stale at 42% — statusline only writes while
rendering; trust the user's number).

## Done

- uv project (`python -m uv`, NOT bare `uv`), deps: numpy, opencv, scipy.
- `farmclip/` package: court model (court.py), video IO (video.py), scene JSON
  writer (scene.py, conforms to docs/specs/scene-format.md), CLI (cli.py).
- out/clip.mp4 = first minute, 1280x720@30. out/debug/ has all overlays.
- Calibration pipeline (fully automatic):
  1. lines.py: white-segment detector (tophat+Hough+merge).
  2. hypothesis.py: RANSAC assignment search. Anchors: near attack line +
     net band (auto-picked). Enumerates sideline/far-line assignments,
     4-point planar PnP via solvePnPGeneric IPPE **in z=0 plane** (rotate
     world floor y=0 -> z=0, map pose back, keep camera-above-floor branch).
     KEY: image-left = world +Z (chirality). Gates: reproj err, camera height
     1.5-12m, |x| behind end line, far-floor-below-net-band-on-screen,
     net-band alignment <40px.
  3. refine_lsq.py: scipy least_squares polish over (f, rvec, tvec, net_h,
     post_hw); floor lines regulation-fixed, net rig geometry free (this gym's
     rig differs from FIVB nominal). Post bases weighted 5x (soft_l1 would
     otherwise mute them).
- Reference frame: 1770 (serve moment). Best overlay: net band + near attack
  line hug within ~10-20px; posts within ~20px; net_h/post_hw hit bounds
  (2.55/5.53) — geometry partially inconsistent, ~25px RMS. ACCEPTED as v0;
  precision re-polish deferred (see known-issues).
- out/calib.json = current reference calibration (includes net_h/post_hw).
- Camera MOVES (45-96px between 10s samples) — per-frame pose propagation
  needed (background feature track -> chained homography). NOT built yet.

## Next (in order)

1. Ball stage: clone/eval asigatchov/fast-volleyball-tracking-inference (ONNX
   CPU, HF: asigatchov/fast-volleyball-tracking) vs YOLO (ovml) on the clip.
   Overlay detections -> pick winner -> ballistic 3D lift between touches.
2. Players v1: person detect/track (OpenCV HOG or a small ONNX detector) ->
   feet -> ground positions via calib homography -> scene JSON.
3. Pose propagation for camera motion (LK track on background, chain
   homographies from frame 1770).
4. Wire all into cli.py -> out/scene.json -> sample for the viewer session.
5. Final overlay verification pass; update roadmap Done section.

## Gotchas

- PowerShell mangles quotes in `python -c` — use scripts/*.py files.
- usage.json needs encoding='utf-8-sig' (BOM).
- cv2.imwrite silently fails if the directory doesn't exist.
- Court chirality: for a +X camera, image-left = world +Z. The mirror-ghost
  planar PnP solutions put the camera below the floor — filter them.

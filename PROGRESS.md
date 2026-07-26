# Tuning campaign state (2026-07-26 ~14:30 ET)

USER TARGETS: ball >=90% of trackable time; all players (12 menlo / 4 mikasa)
on court >=80% of frames. Measure: scripts/metrics.py OUTDIR N.

## Ball (menlo bench, out/): coverage-vs-physics tradeoff matrix
- dual-model union (Grid+V1c): 85% raw detection (was 80 single) ✓ adopted
- grow-by-fit segmentation (replaces accel splitter) ✓ adopted — huge win
- min_len5/rms14/bridge0.7s: 92.8% raw-det coverage BUT speed spikes to 700m/s
- min_len6/rms12 + outlier-reject + tolerant bridges: 73.4%, p99 61 (best clean)
- CONCLUSION: threshold knobs exhausted; next lever = junction-aware estimator
  (segments must agree on position at touches; solve jointly, or spline-smooth
  across touch with position continuity constraint). Also: the 350-700 m/s max
  spikes are single junctions between two long fits — snap adjacent segment
  endpoints to their midpoint before dense emission would kill them cheaply.

## Players
- menlo (12 expected): >=12 in 23.6% frames (target 80). Distribution peaks 7-9:
  far-side misses. Levers untried: yolo11s (bigger model), imgsz 1280,
  per-region conf, ByteTrack-style low-conf second pass.
- coasting probation(5 hits) + margin 1.0 + dedupe 0.4m: adopted, ghosts gone.
- mikasa (4 expected): re-run finished with regulation-locked calib (VERIFIED
  GOOD overlay); check scripts/metrics.py out/mikasa 4.
- mikasa sample cut hit VideoWriter codec error (libopenh264) — check
  examples/samples/mikasa-*/clip.mp4 exist/valid; fallback fourcc mp4v.

## Next iteration order
1. Ball: snap junction endpoints to midpoint (cheap, kills teleports) then
   re-measure the min_len5/rms14 config — likely >=90 clean.
2. Players menlo: try yolo11s + imgsz 1280 on 300-frame slice, measure count
   distribution before full run.
3. Full re-runs both videos -> metrics -> samples -> commit.

# Older note (pre-pause)

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

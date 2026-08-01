# Plan: AI-based automatic court calibration

Goal: stop hand-labeling courts. A fine-tuned keypoint model detects the named
court keypoints per frame; the existing solver stack (`calibrate.solve` →
`calib_solve.py` LM polish → `calib_refine.py` RANSAC) turns them into a pose.
Acceptance stays the same: `calib_eval.py` COURT ≤5px / NET ≤8px medians +
visual overlay in `out/debug/`.

## Why this approach (Phase 0 findings, 2026-07-28)

- No public volleyball calibration model with weights exists. Soccer models
  (PnLCalib, TVCalib) have weights but 57 soccer-specific keypoints — pipeline
  design is reusable, weights are not.
- ~1,600 free CC-BY-4.0 labeled volleyball court frames exist on Roboflow
  Universe (primaryws 862 imgs, volleyballcourt 495, tipe-volley 277) — but
  they cover floor points only, **not** net-top/antenna/post points (which our
  solver needs and which are our best-conditioned points).
- Ultralytics (yolo11s.pt already in repo, ultralytics 8.4 + torch installed)
  fine-tunes a pose model on ~100–800 images; export to ONNX for CPU inference
  (onnxruntime already a dep). Evidence: asigatchov/Court-Keypoint-Detection
  did exactly this for volleyball back-views (500–1000 imgs → robust);
  sportsfield_release generalized from 170.
- Our clicker (`calib.html`) already produces exactly the training labels:
  named 2D points per frame (`annotations.json`). Every clip we hand-label
  from now on grows the training set — labeling stops being wasted work.
- **Label multiplier:** camera is stationary(ish), so one annotated ref frame
  approximately labels every frame of that clip. Sample N frames per clip,
  reuse the annotation (optionally jitter-check with `calib_eval.match_segs`),
  and 2 annotated clips become ~100+ training images with net/antenna points.

Rejected: zero-shot VLM pointing (Molmo) — not CPU-practical, few-px accuracy;
usable later only as an auto-labeler. Classical Hough-only — already tried,
distractor lines defeat it (that's how we got here).

## Keypoint schema

Use the existing 18 names from `farmclip/court.py:21-40` verbatim as the YOLO
pose keypoint order (one "court" object class, 18 kpts, visibility flag 0/1/2
— out-of-frame corners are handled natively by visibility=0). Roboflow floor
points map onto the matching subset; net/post/antenna points come only from
our own labels (visibility=0 in Roboflow images).

**Left/right convention (the bug that burned us):** labels are in WORLD frame
(`_left` = world +Z = image-left for the canonical PoV). Document this in the
dataset config; `calib_solve.py`'s swap enumeration stays as a safety net.

## Phase 1 — Dataset builder (`scripts/build_kp_dataset.py`)

What: convert `annotations.json` + video → YOLO pose dataset.

- For each run dir with an `annotations.json` (menlo `out/`, mikasa
  `out/mikasa/`): sample ~40 frames spread over the clip (copy the frame
  iteration from `farmclip/cli.py:calibrate`, ~line 27), write images +
  YOLO pose label txt per frame reusing the ref-frame annotation.
- Guard against camera drift: score each sampled frame with
  `scripts/calib_eval.py:score_frame` against the run's `calib.json`; drop
  frames whose court median exceeds ~2× the ref frame's (camera moved).
- Bounding box = bbox of visible keypoints padded 5%. Kpt order = the dict
  order of `court.KEYPOINTS`; missing names → visibility 0.
- Emit `data/dataset/court/dataset.yaml` (ultralytics pose format:
  `kpt_shape: [18, 3]`, single class `court`).
- Format reference: asigatchov `coco2yolo_keypoints.py`
  (github.com/asigatchov/Court-Keypoint-Detection) and ultralytics pose docs.

Verify: `python -m uv run python scripts/build_kp_dataset.py` produces N imgs +
N labels; spot-check 3 with a quick draw script (reuse
`calibrate.draw_overlay`-style plotting); ultralytics `check_dataset` loads it.

Anti-patterns: don't invent a COCO intermediate; go straight to YOLO txt.
Don't include frames from clips whose calib never passed visual inspection.

## Phase 2 — Merge public data (optional but cheap)

What: download primaryws (862) + volleyballcourt (495) from Roboflow Universe
in YOLO format, remap their floor keypoints to our 18-slot schema (unmatched
slots → visibility 0), append to the dataset.

- MUST inspect their actual keypoint definitions first (research gap: schemas
  unverified). If remapping is ambiguous, skip the dataset — our own
  multiplied labels may suffice.

Verify: merged dataset.yaml still loads; visual spot-check 3 remapped images.

## Phase 3 — Fine-tune + export

What: `yolo pose train model=yolo11s-pose.pt data=data/dataset/court/dataset.yaml
epochs=100 imgsz=960` (Colab GPU if CPU is painful — dataset is small), then
`model.export(format="onnx")` → commit weights as `finetune_out/yolo11s-court.pt`
+ ONNX next to the ball models' convention.

Verify: val PCK / pose mAP from ultralytics output; run inference on 5 held-out
frames from each clip, draw predicted keypoints, eyeball in `out/debug/`.

Anti-patterns: don't train from scratch; start from yolo11s-pose.pt. Don't
tune hyperparameters before seeing baseline results.

## Phase 4 — Wire into the pipeline (`farmclip/kp_detect.py` + cli hook)

What: inference → named points → existing solver.

- `detect_keypoints(frame) -> dict[str, (u, v)]`: run ONNX model, keep kpts
  with confidence ≥ threshold, return the same shape `annotations.json` has.
- In `farmclip/cli.py:calibrate`: run detection on ~10 sampled frames, take
  per-keypoint median (stationary camera ⇒ medians kill jitter/occlusion),
  feed to `farmclip/calibrate.py:solve` (needs ≥5 pts), then the LM polish
  from `scripts/calib_solve.py:70-86` (lift it into `calibrate.py` so both
  paths share it). Keep the old Hough hypothesis search as fallback when <5
  confident keypoints.
- Optional precision bump if needed: TennisCourtDetector's trick — refine each
  floor keypoint by local crop + line intersection (their README §postprocess)
  before solving.

Verify: `python -m uv run farmclip <clip> <out>` with NO annotations.json
present produces calib.json; `calib_eval.py` medians vs. the manual-annotation
baselines (menlo 31.9px / mikasa COURT 8.2 NET 6.1); overlay jpg inspected.

Anti-patterns: don't bypass `calibrate.solve`'s focal grid; don't add a second
projection helper (three copies exist already — reuse `calibrate.project`).

## Phase 5 — Self-training loop (only if Phase 4 misses goals)

- New clip fails goals → clicker-label its ref frame (existing workflow) →
  rerun Phase 1 builder (it picks up the new run automatically) → retrain.
  Each failure costs one frame of labeling, permanently.
- If keypoint precision is the blocker (overlay close but >5px): chain the
  existing `calib_refine.py:refine_ransac` after the PnP solve — that IS the
  PnLCalib keypoints-then-lines architecture, already built.

## Final phase — Verification

- Both clips: `calib_eval.py` COURT ≤5 / NET ≤8, or explicitly record the gap
  in docs/plans/calibration-goal.md with per-line breakdown.
- Overlay videos (`scripts/overlay_video.py`) for both clips inspected.
- Grep guard: no new `cv2.solvePnP` call sites outside calibrate/hypothesis/
  line_icp; no duplicated projection helper.
- One runnable check: tiny `test_kp_dataset.py` asserting the builder emits
  valid YOLO rows (18 kpts, visibility ∈ {0,1,2}, coords normalized).

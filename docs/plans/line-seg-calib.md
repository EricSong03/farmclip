# Plan: court-line segmentation calibration (v2 model)

Goal: replace/augment keypoint regression with per-line semantic segmentation
for better accuracy + unseen-venue generalization. Pattern follows PnLCalib
(soccer): segment line identities → geometric solve from whole lines.

## Classes (9)

0 background, then: sideline_left, sideline_right, endline_far, endline_near,
attack_far, attack_near, center_line, net_band. Left/right/near/far in WORLD
frame, consistent by construction (masks are rendered from solved calibs).

## Phase 1 — mask dataset builder (`scripts/build_lineseg_dataset.py`)

Labels come FREE from existing work: every labeled frame already has a solved
calib (video runs: calib.json × sampled frames; web images: solve_web from
clicks; dome runs: calib.json). Render each court line through the calib as a
~5px-wide polyline into a class-index PNG mask alongside the image.
Reuses build_kp_dataset's frame sampling + drift gate. Output:
`out/finetune/lineseg/{images,masks}/{train,val}` + meta.json (class names).

## Phase 2 — model + training (`scripts/train_lineseg.py`)

Small UNet (torch, ~2M params, encoder 4 levels) → K-class logits at 1/2 res.
Loss: weighted CE (background ↓, lines ↑) + dice. imgsz 960. AdamW 3e-4,
~200 epochs on GPU (user's box; CPU smoke-test flag `--steps N`).
YOLO-seg rejected: instance masks are poor for 3px-wide lines.

## Phase 3 — inference + line solve (`farmclip/lineseg.py`)

Per class: threshold logits → skeletonize/sample pixels → the solve optimizes
(rvec, tvec, f, net_h) minimizing point-to-projected-line distance (same
machinery as solve_web's joint refine, but dense). Init from keypoint model or
hypothesis search. Chain into cli.calibrate as highest-priority path once it
beats the keypoint path on calib_eval.

## Phase 4 — eval

Same gates: calib_eval medians (menlo/mikasa/domes), kp_eval-style gallery for
line masks, unseen-venue check on testimgs. Ship only if it beats v4 keypoints.

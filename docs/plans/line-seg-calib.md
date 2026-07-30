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

## Phase 3 — inference + line solve (`farmclip/lineseg.py`) — DONE

Built as planned, with three deviations worth recording:

- **Mean softmax over the sampled frames, then argmax** (not per-frame masks).
  Players occlude different lines in different frames; the camera is
  stationary, so averaging first is strictly cleaner and costs one pass.
- **Interpretation-plane residuals**, not projected endpoints:
  `l = K⁻ᵀ (a_cam × b_cam)`, distance = `l·[u,v,1] / ‖l₁₂‖`. Projecting the
  endpoints breaks whenever one is behind the camera — constant for sidelines.
  `soft_l1` loss, since segmentation always leaks a few pixels onto other
  sports' floor markings. Each class is weighted `1/√n` so a long sideline
  can't outvote a short attack line.
- **Net height is withheld when unobservable.** A camera in the net plane
  (x=0) projects EVERY net height to the same image line — and sideline
  cameras sit near that plane. The solve perturbs net_h by ±0.15 m post-hoc
  and drops `net_h_est` if the residual barely moves. Pose is unaffected;
  the reported height would have been fiction. Guarded in `_self_check()`.

Wiring (`cli.calibrate`): keypoints → lineseg refine → hough → lineseg refine.
The keypoint calib now seeds lineseg **even when the 12px gate rejects it**
(`rejected=True`) — a sloppy pose is still a fine init, and dense line
evidence is what pulls it onto the paint. Accepted at ≤6px median line error.

Run the solver's self-check with `python -m farmclip.lineseg`.

## Phase 4 — eval

## Phase 4 — eval

Same gates: calib_eval medians (menlo/mikasa/domes), kp_eval-style gallery for
line masks, unseen-venue check on testimgs. Ship only if it beats v4 keypoints.

v2 weights (200 epochs) hit per-class val IoU 0.64–0.73 on 5px-wide lines.
Debug output per run: `debug/lineseg_mask.jpg` (segmented pixels) next to the
usual `calib_overlay.jpg`.

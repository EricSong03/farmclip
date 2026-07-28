# ICRN H200 fine-tune — results note (2026-07-28)

Runbook: [icrn-finetune.md](icrn-finetune.md). Box: ICRN H200 NVL, repo copy at
`~/farmclip` with `videos/`, `scripts/`, `farmclip/`, `docs/` uploaded.

## Task 1 — player detector: DONE

Artifact: **`finetune_out/yolo11s-vb.pt`** (copied from
`runs/detect/finetune_out/yolo11s-vb/weights/best.pt` — ultralytics resolves a
relative `project=` under its own `runs/` dir, so `gpu_finetune.py`'s
`shutil.copy` at the end of `players-train` failed; that copy is the only step
of the runbook that did not run itself).

- **Step 0 gate passed**: yolo11x @ 1536 does see the far-side net-occluded
  menlo players (`out/debug/step0_teacher/`, 25-30 persons/frame incl. the
  far-side row behind the net band). So the Roboflow merge was not required.
- Data: teacher pseudo-labels every 15th frame of both clips → 2484 train /
  276 val images. Val split is **time-disjoint** (last 10% of each clip,
  `gpu_finetune.py` change vs the committed version, which used a random 10% —
  frame-adjacent leakage would have inflated it).
- Train: yolo11s @ 1280, 60 epochs, ~76 min.

### Metrics (val = teacher labels, so "before" is flattered — stock and teacher
are the same family and share blind spots)

| set | model | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| all val (276) | stock yolo11s | 0.984 | 0.879 | 0.974 | 0.935 |
| all val (276) | **finetuned** | **0.991** | **0.923** | 0.979 | 0.960 |
| menlo-only val (12) | stock yolo11s | 0.926 | 0.711 | 0.905 | 0.833 |
| menlo-only val (12) | **finetuned** | **0.970** | **0.820** | 0.948 | 0.915 |

The all-val row is dominated by mikasa (264 of 276 frames, 2v2, easy). The
menlo-only row is the one that matters — the hard far-side/net-occlusion class:
**recall 0.833 → 0.915, mAP50-95 0.711 → 0.820**, i.e. the student recovers
roughly half the gap between stock @1280 and the teacher @1536, at CPU-runtime
cost. Visual before/after: `out/debug/players_eval/` (blue = stock, red = ft),
`scripts/players_eval.py`.

Raw box counts on menlo val: stock 323 → ft 407 (+84 over 12 frames). Zoom
crops show most of that surplus is **background/bleacher people**, not on-court
players — expected (the teacher at 1536 labels them too), and harmless because
the pipeline drops anything whose floor-ray lands off court.

### Pipeline-level check attempted, NOT valid — calibration failed here

`scripts/players_metric.py` runs the real players stage (YOLO → tracker →
floor-ray → court filter → roster cap) and reports the `metrics.py` number
(% of player-frames with ≥12 on-court). It printed stock 28.7% → ft 31.2%, but
**that comparison is worthless**: auto-calibration on the uploaded
`videos/clip.mp4` converged to a bad pose (7 inliers, `out/players_metric/clip/
debug/calib_overlay.jpg` shows the court model projected several metres toward
the camera). Consequence: essentially every real player falls in the x<0 half —
drop-reason census over 600 frames was 30959 tracks → 2527 edge-clipped, 19419
off-court, 9013 on-court, with per-frame side counts like (teamA 0, teamB 9).
To re-measure this properly, upload the laptop's good `out/calib.json` (or
re-run calibration locally) — the detector change cannot be judged through a
broken pose.

## Task 2 — ball model: NOT DONE HERE (2026-07-28: user is running it elsewhere)

Nothing ran. It was blocked on inputs anyway; recorded here in case the work
comes back to this box. Missing from the upload:

1. **`out/finetune/ball/*.csv`** — the label set the whole task exists to train
   on. The directory does not exist on this box. Hard blocker.
2. **`vendor/fast-vb-tracking/models/*.onnx`** — needed as the numerical
   reference for the export check (`VballNetV1c_seq9_grayscale_best.onnx`) and
   as the checkpoint to fine-tune from. Not uploaded.

Available/verified here: network works (`git clone` of
`github.com/asigatchov/vball-net` succeeded — TensorFlow/Keras trainer,
`src/train_v1.py`, TrackNet-style video+csv dataset layout, matching our csv
format), both source videos, H200 idle. The repo's README points at a Google
Drive / Yandex demo-model bundle, but that is *not* obviously the V1c seq9
grayscale checkpoint farmclip runs, so uploading the local one is safer.

## Surprises worth carrying back

- The player metric is no longer detection-limited in the way the runbook
  assumed. Even with a perfect detector, the census above shows the pipeline
  discards ~2.5k tracks/600 frames on the frame-edge rule and the remainder is
  gated by calibration quality and the `per_side` cap. Fixing calibration
  robustness on new clips likely moves "all 12" more than any further detector
  work.
- Teacher-label val scores flatter the stock baseline (shared blind spots).
  Menlo-only + visual inspection is the honest read; whole-val mAP is not.

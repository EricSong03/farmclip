# court-pose v6 results (GPU round, 2026-07-31)

Ran the handoff in `docs/icrn-training.md` on an H200 box. Shipped model is
**court-pose-v6b**; the doc's literal 300-epoch recipe (v6) produced a model
with a dead output channel and is kept only as a record.

## What shipped

| artifact | what it is |
| --- | --- |
| `finetune_out/court-pose-v6b/weights/best.pt` | keypoints, 700-epoch schedule, best @ epoch 343 |
| `finetune_out/yolo11s-court.onnx` | v6b exported at imgsz=1280 — the file the pipeline loads |
| `finetune_out/lineseg/{best.pt,lineseg.onnx}` | lineseg retrained on refreshed masks |
| `finetune_out/court-pose-v6/{results.csv,args.yaml}` | the failed 300-epoch run, weights deliberately not committed |

## Keypoints: v5 vs v6 vs v6b

Val split is the corrected one (60 images), so v5 is charged for its inverted
handedness. "L/R-swap" scores each image against both the labels and their
mirror and keeps the better — that separates *wrong place* from *right place,
wrong name*.

| | no detection | median as-labelled | median best-of-swap | images preferring the mirror |
| --- | --- | --- | --- | --- |
| v5 | 13/60 (22%) | 697.1px | 17.9px | 46 of 47 |
| v6 (300 ep) | 1/60 (2%) | 27.8px | 27.8px | 3 of 59 |
| **v6b (700 ep)** | **3/60 (5%)** | **13.7px** | 13.7px | 3 of 59 |

v6b PCK@0.02 of image diagonal: 76.7% (v6: 60.1%). Worst remaining keypoint is
`corner_near_left` at 109.7px on n=9 — it is labelled in only 41 images total,
so it is a data-volume problem, not a model one.

The handedness fix works. v5's geometry was never wrong; its names were.

## Why the doc's 300-epoch recipe failed

v6 trained cleanly by every loss curve, then failed exactly two output
channels: `attack_near_right` (571px median on **train**, 570px on val) and
`corner_near_left` (393/372px), while its other 16 keypoints sat at 9-36px.

It is not the labels. Fitting a homography from the *other* floor points and
reprojecting reproduces the `attack_near_right` label to **6.2px median across
all 241 images** — the labels are geometrically self-consistent. Feeding the
model a horizontally flipped image moves the failure to whichever name lands in
that slot, so the defect follows the **raw output channel**, not the world
point. The base checkpoint has no such channel: it predicts both near-attack
points accurately, just mirrored.

So: unlearning the old inverted handedness stranded two channels in a local
minimum, and 300 epochs with `patience=50` (best @ 86, stopped @ 136) ended
before they escaped. v5's own schedule — `epochs=700 patience=250` — fixes it
(best @ 343, stopped @ 593). **Use v5's schedule, not the doc's 300/50.**

This mattered downstream, which is how it was caught: with v6 the pipeline's AI
keypoint anchor was *rejected* on both local clips (96.0px floor error on
menlo, 114.2px on mikasa, both over the 12px gate) and fell back to Hough. With
v6b it is accepted at 6.9px and 5.2px.

## Lineseg refresh: a wash, committed anyway

Masks were regenerated **web-only**. The video runs could not be redone on this
box — `out/runs.json` is absent and 4 of the 6 run videos are not in the repo —
so their masks are unchanged and still carry the 15-30px calib error the
handoff complains about. The web half went from 62 stale images to **116
freshly solved venues** (median floor err 3.54px, p90 7.4px, max 16.1px; 6 of
122 labels rejected over the 30px gate). Dataset is now 228 train / 56 val.

| | val loss | mIoU (fg) |
| --- | --- | --- |
| v2 | 0.7519 | 0.590 |
| v3 (this round) | **0.6706** | 0.574 |

Effect on the actual calibration, scored with `calib_score` coverage on a clean
frame: mikasa coverage 0.488 -> 0.503 and err 6.14 -> 5.73px, menlo 0.324 ->
0.317 and 13.82 -> 14.32px. A wash. Committed because the provenance is
strictly better, not because it measurably helps yet.

Note both clips still **fail** the coverage verdict (menlo 2/7 lines supported,
mikasa 3/7, both under the 0.6 mean-coverage gate) even where the mikasa
overlay looks visibly correct — the gate is stricter than these two glossy-wood
gyms can satisfy. Do not read those FAILs as "the calib is wrong"; read the
overlay.

## Environment notes for the next run

- The default `torch` wheel now needs a newer driver than this box's 12.8.
  Pin `torch==2.10.0+cu128` / `torchvision==0.25.0+cu128` from the pytorch
  cu128 index after `uv sync`, or CUDA init fails outright.
- `out/finetune/court/dataset.yaml` has a Windows absolute `path:`. Left as is;
  training used a sibling `dataset.linux.yaml` (untracked) so the tracked file
  stays correct for the Windows box.
- Ultralytics writes runs to `runs/pose/finetune_out/<name>/`, not
  `finetune_out/<name>/`. Artifacts were copied across by hand, same as v5.

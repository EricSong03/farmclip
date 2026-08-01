# GPU training handoff (icrn session)

> **This round is done — see `docs/plans/court-pose-v6-results.md`.**
> Shipped: **court-pose-v6b** (13.7px median, handedness correct, 5% no-detect)
> plus a lineseg refresh. One correction to the recipe below: the
> `epochs=300 patience=50` schedule **stranded two output channels dead**
> (`attack_near_right`, `corner_near_left`) while unlearning v5's inverted
> handedness, which made the pipeline reject the AI anchor on both local clips.
> v5's own `epochs=700 patience=250` fixes it. Use that.

Context: farmclip auto-calibrates volleyball footage from named court
keypoints. Two models:

- **court-pose keypoints** (yolo11s-pose, 18 pts) — v5 deployed. Excellent on
  venues it has seen (dome venues 0.7-6.7px, beating hand-clicked anchors) but
  thin on unseen ones: 3/12 unseen web images detected, and **zero keypoints**
  on a head-on clip. Widening that distribution is this round's whole point.
- **lineseg UNet** (9 classes: 7 floor lines + net band + bg) — v2 trained and
  wired in. Correct on mikasa, wrong on most other venues. It is **downstream**
  of the keypoints: its training masks are rendered *through* solved calibs, so
  it can never be better than the calibs that trained it.

## This round's primary job: keypoints v6

Keypoints first, not lineseg, precisely because lineseg is a student of the
calibs. Better clicks -> better keypoints -> better masks -> better lineseg.

```
yolo pose train model=finetune_out/yolo11s-court.pt data=data/dataset/court/dataset.yaml \
  epochs=300 imgsz=1280 batch=16 optimizer=AdamW lr0=0.0003 mosaic=0 \
  scale=0.3 translate=0.05 patience=50 device=0 project=finetune_out name=court-pose-v6
```

Rules that exist for hard-won reasons: **mosaic=0 always** (NaN crashes on this
data), keep `imgsz=1280` (matches the deployed ONNX), AdamW `lr0=3e-4`.
Fix `data/dataset/court/dataset.yaml`'s absolute `path:` if the repo lives
somewhere else on the GPU box.

Export when done:

```
yolo export model=finetune_out/court-pose-v6/weights/best.pt format=onnx imgsz=1280
```

## What changed in the data since v5 - read this before judging results

The dataset roughly doubled AND one systematic label bug was fixed. Both
matter for interpreting v6 vs v5.

1. **Left/right inversion, fixed.** `consistent_names()` renames clicks to
   match the solved pose, and because the court is symmetric the solve is free
   to pick the mirrored world assignment - it did so on **51 of 52** web
   images, so every `_left` the human clicked was stored as `_right`. v5 was
   trained on those inverted labels. `build_kp_dataset.py` now applies
   `calibrate.canonical_lr()` after `consistent_names`, anchoring handedness on
   a single-depth reference pair (net/antenna/centre), so `_left` always means
   image-left. **v6 is the first model trained on correct handedness**, so a
   v5-vs-v6 comparison is not apples to apples on any left/right metric.

2. **Distribution widened deliberately.** v5's set was dominated by elevated
   sideline views of large arenas. Added since: ~70 images from Wikimedia
   Commons plus amateur YouTube VODs (school, club, rec, intramural), chosen
   for camera *placement* variety - floor-level, behind-the-endline head-on,
   corner/diagonal, and small gyms with basketball/badminton lines painted
   through the court. Pro broadcast footage was deliberately excluded: it is
   effectively one camera angle and deepens the existing bias.

3. **Solver fixes that improved the labels feeding training.** `solve_web`'s
   joint net refine could destroy a good floor solve (measured: a 0.6px floor
   fit came back at 346px with net height pinned to a bound). It now keeps the
   floor pose unless the refine holds up, and measures net height with the pose
   frozen. Across 107 well-spread labelled images: median floor error **3.5px**,
   100 under 10px, and net height independently measured on 100 venues with a
   **median of 2.44 m** against a 2.43 m regulation net.

## Optional second job: refresh lineseg

Only worth running after v6 exists, and only after regenerating its masks -
they are rendered through solved calibs and inherit calib error. The current
masks include venues whose calibs were 15-30px off (menlo 30.3px, dome4
15.7px), which is why that model learned misplaced lines.

```
python -m uv run python scripts/build_lineseg_dataset.py     # regenerate masks first
python -m uv run python scripts/train_lineseg.py --device cuda --batch 8 --epochs 200
```

Checkpoints `finetune_out/lineseg/{last,best}.pt` (best = lowest val loss).
~2 min/epoch at 960px. Per-class IoU prints each epoch; net_band is thin and
lags early, that is normal.

## Bring back

```
git add -f finetune_out/court-pose-v6/weights/best.pt
git add -f finetune_out/yolo11s-court.onnx                  # if you exported
git add finetune_out/court-pose-v6/results.csv
git add -f finetune_out/lineseg/best.pt                     # if lineseg re-ran
git commit -m "feat(calib): court-pose-v6 weights" && git push
```

(`-f` needed: `*.pt` is gitignored.)

## How to judge the result

Do **not** rank by reprojection error alone - it is gameable by sparsity and
has already fooled us: a visibly wrong pose scored 1.34px while a correct one
scored 1.89px, because fewer/shorter pieces of evidence fit any pose better.
Use `farmclip/calib_score.py`, which scores *coverage* - what fraction of each
projected line's length actually has image evidence - alongside error:

```
python -m uv run python -m farmclip.calib_score     # self-check
```

The real acceptance test is still the overlay: run the pipeline on a clip and
look at `out/<run>/debug/calib_overlay.jpg` before calling anything done.

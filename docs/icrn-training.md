# GPU training handoff (icrn session)

Context: farmclip auto-calibrates volleyball footage. Two models:
- **v1 keypoints** (yolo11s-pose, 18 pts) — trained, deployed, works on seen
  venues (mikasa 5.6/4.3px, beats manual) but hallucinates on unseen gyms.
- **v2 line segmentation** (custom UNet, 9 classes: 7 floor lines + net band
  + bg) — NEW, untrained. The bet: per-pixel line identity generalizes across
  venues where point regression doesn't. See docs/plans/line-seg-calib.md.

Datasets in this repo are FRESH (all 49 web images + 6 video runs, ~30 venues):
- `out/finetune/lineseg/` — 174 train / 43 val images+masks (v2)
- `out/finetune/court/`  — 188 train / 46 val YOLO pose (v1)

## Primary job: train v2

```
python -m uv run python scripts/train_lineseg.py --device cuda --batch 8 --epochs 200
```

- Checkpoints: `finetune_out/lineseg/{last,best}.pt` (best = lowest val loss).
- Healthy run: val loss falls steadily; per-class IoU printed per epoch —
  floor-line classes should climb well above 0.3; if any class stays ~0 after
  50 epochs, note it (net_band is thin and lags — that's normal early).
- If val plateaus early, extend: rerun with `--epochs 400` (resumes from
  last.pt if you pass `--resume finetune_out/lineseg/last.pt` — check the
  script's flags with `--help`; if no resume flag exists, just train longer
  from scratch, it's a small model).
- ~2 min/epoch on a modest GPU at 960px.

## Optional second job: refresh v1 (dataset doubled since its last train)

```
yolo pose train model=finetune_out/yolo11s-court.pt data=out/finetune/court/dataset.yaml \
  epochs=300 imgsz=1280 batch=16 optimizer=AdamW lr0=0.0003 mosaic=0 \
  scale=0.3 translate=0.05 patience=50 device=0 project=finetune_out name=court-pose-v5
```

Rules that exist for hard-won reasons: **mosaic=0 always** (NaN crashes on
this data), keep imgsz=1280 (matches deployed ONNX), AdamW lr0=3e-4.
Fix `out/finetune/court/dataset.yaml`'s absolute `path:` if the repo lives at
a different path on the GPU box.

## Bring back

```
git add -f finetune_out/lineseg/best.pt            # required (v2)
git add -f finetune_out/court-pose-v5/weights/best.pt   # if v1 refresh ran
git add finetune_out/lineseg/*.csv finetune_out/court-pose-v5/results.csv 2>/dev/null
git commit -m "feat(calib): v2 lineseg weights [+ v1 refresh]" && git push
```

(`-f` needed: *.pt is gitignored.) The local session then builds v2 inference
+ line-based solve (Phase 3 of the plan) and benchmarks v2 vs v1 on
calib_eval — winner becomes the pipeline's first-choice path.

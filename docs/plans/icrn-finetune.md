# ICRN H200 fine-tune session — runbook for the Claude session driving it

You are running a one-time GPU fine-tune for **farmclip**, a pipeline that turns
single stationary-PoV volleyball video into 3D game models. Runtime inference is
CPU-only; your job is to produce two model artifacts that lift detection quality,
then hand them back. Train once, run on CPU forever.

## Why (current CPU-tuned scoreboard, targets in parens)

| metric | menlo 6v6 720p | mikasa 2v2 60fps | target |
|---|---|---|---|
| ball % of in-play time | 69.6 | 81.5 | (90) |
| all players on court | 1.3% | 20.5% | (80%) |

Blockers are model-side, not threshold-side: generic COCO yolo11s misses small
net-occluded far-side players; the pretrained VballNet ensemble misses ~15% of
raw ball frames (spike blur, net crossings). Threshold tuning is exhausted.

## Inputs to upload (from the local repo)

- `scripts/gpu_finetune.py` — player-detector stages (self-contained).
- `videos/` — the source clips: `out/clip.mp4` (menlo, 1 min 720p30) and the
  mikasa vod first ~11 min re-encoded to mp4 (any name, `.mp4` extension).
- `out/finetune/ball/*.csv` — ball labels (`Frame,Visibility,X,Y`, one csv per
  clip, produced locally by `scripts/export_finetune_data.py`). Three classes:
  ensemble-consensus, ballistic-fit-backed single-model detections, and
  gap-interpolated frames inside accepted fits where BOTH models missed —
  that last class is the spike-blur / net-crossing frames this run must fix.

## Task 1 — player detector (fully scripted)

**Step 0 — gate the teacher before training.** Run yolo11x @ 1536 on ~5 menlo
frames that show far-side players behind the net and render overlays. If the
teacher also misses them, pseudo-labels inherit the blindness and the student
learns "nothing there" — in that case the Roboflow merge below is REQUIRED,
not optional, before training.

```
pip install ultralytics opencv-python-headless
python gpu_finetune.py players-data --videos videos/
python gpu_finetune.py players-train
```

- Teacher = yolo11x @ imgsz 1536 pseudo-labels every 15th frame; student =
  yolo11s @ 1280. Deliverable: `finetune_out/yolo11s-vb.pt`.
- Optional boost if val mAP is weak: merge a Roboflow Universe volleyball
  players dataset (e.g. "Tracking Volleyball Players") into `player_dataset/`
  before training — same YOLO layout.
- **Acceptance:** on ~20 held-out menlo val images, the student should detect
  ≥11 of 12 players in most frames (teacher labels are the reference). Render a
  few prediction overlays and look at them; far-side players behind the net are
  the class that must improve.

## Task 2 — ball model (adapt the vendor trainer)

Fine-tune VballNet from its pretrained checkpoint on our labels.

1. `git clone https://github.com/asigatchov/vball-net` and read its README /
   training entrypoint. (Local repo has inference-only sibling
   `fast-vb-tracking`; production checkpoint we fine-tune from:
   `VballNetV1c_seq9_grayscale_best` — grayscale, seq9, 288x512 input.)
2. Arrange our data in its expected dataset layout (TrackNet-style: per-clip
   video + `Frame,Visibility,X,Y` csv — our csvs are already in that format;
   frames with `Visibility=0` are unlabeled, exclude or mask them, they are NOT
   confirmed ball-absent). **fps note:** menlo is 30fps, mikasa is 60fps, and
   VballNet stacks 9 consecutive frames — build mikasa training sequences at
   frame stride 2 so per-frame ball displacement matches menlo and the
   pretrained checkpoint's motion scale (inference runs at native fps, so keep
   some native-stride mikasa sequences too if the trainer makes that easy).
3. Fine-tune from the pretrained checkpoint (low LR, ~10-20 epochs is the
   TrackNet-family norm for adaptation; a few GPU-hours max).
4. Export ONNX **matching the inference contract** of
   `fast-vb-tracking/src/inference_onnx_seq_gray_v2.py`: same input tensor
   (9-channel grayscale stack, 288x512) and output heatmap as
   `VballNetV1c_seq9_grayscale_best.onnx`. Verify export **numerically**, not
   just by shape: first re-export the UNCHANGED pretrained checkpoint through
   your export path and confirm its heatmap matches the vendor ONNX within
   tolerance (max abs diff < 1e-3) on one real 9-frame input via onnxruntime
   CPU EP. Only then trust the fine-tuned export through the same path.
- **Acceptance:** hold out a TIME-DISJOINT segment (e.g. the last ~90s of
  mikasa), never trained on — random 10% of frames from trained clips mostly
  measures memorization of the same rallies. Fine-tuned model must beat the
  pretrained checkpoint on Acc@5px on that held-out segment, and separately on
  the fit-backed + gap-interpolated label classes (the hard frames).

## Deliverables to hand back

- `finetune_out/yolo11s-vb.pt`
- `vballnet-vb-finetuned.onnx`
- A short metrics note: player val mAP50 (before/after), ball held-out Acc@5px
  (before/after), and anything surprising.

Scope honesty: these artifacts are tuned to these two venues. That's fine —
the current goal is these videos. A new gym may need a repeat of this runbook
with that venue's footage (the export script regenerates labels from any
pipeline run, so the loop is cheap).

## Local wiring after download (for reference, done back on the laptop)

- Players: `farmclip ... --player-model yolo11s-vb.pt`.
- Ball: drop the onnx into `vendor/fast-vb-tracking/models/` and put it first
  in `VBALLNET_MODELS` in `farmclip/cli.py`.
- Re-run both benchmark videos, `scripts/metrics.py out 12` and
  `scripts/metrics.py out/mikasa 4`, compare against the scoreboard above.

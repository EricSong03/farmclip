# Known Issues

## Calibration / pipeline (from the loop build, 2026-07-26)

- **Calibration v0 precision:** reference-frame pose (frame 1770) aligns net band,
  near attack line, and posts within ~20-40px, but the solver pushes net-rig
  params to their bounds (net_h 2.55, post_hw 5.53) — geometry partially
  inconsistent, likely sub-optimal far-court accuracy. Re-polish deferred.
- **Camera drifts** (45-96px between 10s samples) but all stages currently use
  the single reference calib. Per-frame pose propagation (background feature
  tracking → chained homography) is designed but not built. Ball/player world
  positions inherit this drift error away from the reference frame.
- **Ball detector picks up dead-time noise** (held balls, ball-like objects
  between rallies); the ballistic fit's sanity gates reject most of it, but
  out-of-rally frames can still contain garbage 3D points.
- **YOLO11n generic "sports ball" is useless on this footage** (16% detection,
  mostly false locks) — don't revisit without a volleyball-finetuned model.

## Court keypoint pose model (court_train_pack, 2026-07-30)

Trained `yolo11s-court.pt` on the 18-keypoint pack (94 labelled frames:
menlo 40, mikasa 27, web-scraped 27). Runs in `runs/pose/`; eval via
`scripts/court_kp_val.py`, L/R repair via `scripts/fix_lr_labels.py`.

- **The pack shipped with train/val leakage.** `web_web_014` and `web_web_017`
  were byte-identical duplicates present in *both* splits (2 of 19 val images,
  10.5%), inflating every val metric. Quarantined out of train to
  `court/_quarantine/`. Re-check any future pack for this.
- **L/R naming convention is world-anchored, not image-anchored.** `_left`
  keypoints sit at the LARGER image-x in ~97% of frames. 8 label files (all
  `web_web_*`) carried the inverted convention; `scripts/fix_lr_labels.py`
  canonicalises them (per-frame majority vote, then residual per-pair repair).
  Note `fliplr` augmentation is SAFE with the pack's `flip_idx`: mirroring plus
  the L/R rename preserves "left at larger x".
- **4 of 18 keypoints are effectively unlabelled.** `post_top_left` (0 visible
  instances), `post_top_right` (4), `post_base_left` (3), `post_base_right` (3).
  They are masked out of the pose loss, so their outputs are untrained noise.
  Never feed them to solvePnP. `corner_near_right` is also thin (24/94).
- **`antenna_tip_right` needs a LONG schedule to learn at all.** The image-left
  antenna is a thin red/white pole against a dark brick pillar (mikasa) or
  cluttered stands (menlo), and is visually near-identical to its mirror twin,
  so the model sits in the wrong mode for hundreds of epochs. At 300 epochs it
  is confidently wrong (~314px median); at 661 epochs it converges to ~34px.
  Labels are consistent (verified), so this was never a labelling bug. Do not
  judge this keypoint -- or stop training -- before ~450 epochs.
- **300 epochs is not enough; use `epochs=700 patience=250`.** The recipe's
  300/50 stops far too early. v4 was still improving at epoch 452
  (mAP50-95(P) 0.562 -> 0.672) and only early-stopped at 661.
- **Keypoint confidence does NOT predict error — do not filter on it.**
  Pearson r = 0.029 between predicted kpt-conf and pixel error. The *bad*
  predictions score mean conf 1.000, *higher* than the good ones (0.984);
  `antenna_tip_right` is confidently wrong at conf 1.000. Exclude unreliable
  keypoints by index, not by threshold.
- **`imgsz=640` costs precision on 1280x720 footage** but the effect is small
  next to data quantity; 1280 mainly shrinks the error tail (p90 475 -> 357px).
- **Ultralytics `pose mAP50` is near-useless on this val set** — it sat pinned
  at exactly 0.685 for a whole 300-epoch run (19 images, one instance each, and
  uniform OKS sigmas since nkpt != 17). Judge this model by pixel error.
- **Training clips-only makes the core 12 keypoints better but breaks BOTH
  antennas** (532/625px). The 19 heterogeneous web images are what breaks the
  antenna mirror symmetry. Keep them in the training mix.
- **Model comparison on the 13-image clips val (median / mean / p90 / max px):**
  v1 640, dirty labels 17.5 / 81.1 / 332 / 562; v2 1280 300ep 18.2 / 48.5 / 88 /
  362; v3 1280 clips-only 19.2 / 110.8 / 534 / 668; **v4 1280 700ep (all
  sources) 21.8 / 35.9 / 94 / 166 <- ship this one**. v4 has a slightly worse
  median but is the only run with no catastrophically-broken keypoint (worst is
  corner_near_left at 42px). Since kpt-confidence cannot flag outliers, bounded
  worst-case matters far more than best-case median for solvePnP.

## GPU training round 2 (lineseg v2 + keypoints v5, 2026-07-30)

Both trained on the refreshed ~30-venue datasets (lineseg 180/50,
keypoints 197/59 -- slightly larger than docs/icrn-training.md states).

- **v2 lineseg UNet works.** 200 epochs, batch 8, 960x544, 3.35M params.
  Best val loss 0.3981 at epoch 163; all 8 non-bg classes land at 0.64-0.72
  IoU. `net_band` reached 0.68 and did *not* lag the floor lines the way the
  handoff predicted. Val plateaued ~epoch 160, so 200 epochs was about right
  (cosine LR had annealed by then) -- extending to 400 is not obviously worth it.
- **`epochs=300 patience=50` is wrong for keypoints too -- confirmed twice.**
  v5's best epoch was **393** (mAP50-95(P) 0.8228) and it never early-stopped
  across the full 700. The 300/50 recipe in `scripts/train_court.sh` and in
  docs/icrn-training.md would have cut it off before the optimum. Use 700/250.
- **v5 beats v4 by a wide margin on the new 59-image val split**
  (median / mean / p90 px, via `scripts/court_kp_val.py`):
  v4 77.8 / 158.9 / 412.5 (PCK@0.02 30.5%, 518 kpts matched);
  **v5 17.7 / 47.9 / 55.0 (PCK@0.02 86.0%, 630 matched)**. The real gap is
  wider still: 7 of those 59 val images were in v4's *training* set.
- **v5's weak spot is `corner_near_right`** (452px median, but only 11 GT
  instances). Corners generally remain the worst group; the mid-court and net
  keypoints are all at 11-20px median. Same rule as before -- exclude by index,
  not by confidence.
- **`project=finetune_out` does not land in `finetune_out/`.** Ultralytics
  resolves `project` relative to its own `runs_dir`, so results went to
  `runs/pose/finetune_out/court-pose-v5/`. The committed copy under
  `finetune_out/court-pose-v5/` is a manual copy. Pass an absolute path if you
  want the doc's layout.
- **`scripts/train_lineseg.py` has no `--resume` and writes no CSV.** Per-epoch
  val loss and IoU exist only on stdout -- tee it. The committed
  `finetune_out/lineseg/train_log.txt` is that capture.

## Footage / calibration

- **Court corners rarely visible.** Typical PoV is low, behind the end line — the "click 4 corners" flow in the current stub is unusable on real footage. Superseded by named-keypoint spec; stub UI not yet updated.
- **Distractor lines everywhere.** Multi-sport floors (badminton/basketball lines) and adjacent courts defeat naive line detection. Any automatic detector must be trained/scoped to "the court being played on."
- **Pan/zoom videos.** A single calibration won't hold; needs per-N-frame refresh (planned, not built).

## Stub (index.html)

- "Calibrate" only logs points to console — no pose solve yet.
- Corner overlay coordinates are normalized to the video *element*, not the video *frame* — letterboxing from `object-fit: contain` will skew clicks. Fix when real solving lands.
- Net height fixed at 2.43 m (men's); no toggle yet.

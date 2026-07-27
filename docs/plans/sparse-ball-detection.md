# Sparse ball detection (shelved — pick up after physics-first goal is met)

Physics-first needs ~5-8 anchors per flight, not 30-60. Detection (two
VballNet passes, every frame) is ~all the wall-clock; the fit is cheap.
Target: 4-6x faster on clips, more on full games, verified by unchanged
holdout numbers (docs/plans/physics-first-goal.md thresholds).

Constraint that shapes everything: VballNet is seq-9 — one inference eats 9
CONSECUTIVE frames. The skippable unit is the 9-frame chunk, not the frame.

## Layers (cheapest first, each verified before adding the next)

1. **Burst sampling** — process 1 chunk, skip 2 (3x less inference).
   Bursts of 9 consecutive detections every ~0.9s: dense local velocity
   (good for break detection), flights span multiple bursts. Contacts in
   skipped chunks cost ~0.3s cut localization — merge/refit already absorbs
   that. Knob: `--burst 1/N`, try 1/3 then 1/4.
2. **Rally gating** — one chunk every ~2s while idle; first chunk with
   consensus anchors switches to burst mode until no ball for a few seconds.
   ~2x more on full games (dead time between rallies).
3. **ball1 on demand** — run ball0 per layers 1-2; run ball1 only on chunks
   where ball0 fired (consensus needs both, but ball1 is only confirmation).
   Skips 30-40% of the second model's work.

## Where

`farmclip/cli.py run_ball` + the vendor inference wrapper
(`vendor/fast-vb-tracking/src/inference_onnx_seq_gray_v2.py` — needs a
chunk-mask or frame-list argument; today it walks the whole video).

## Acceptance

- physics_eval holdout on both clips within goal thresholds at each layer
  (median ≤3px, p90 ≤12px, coverage ≥85% — no regression vs dense baseline).
- Wall-clock measured per layer, recorded here.

## Do NOT

- Lower video resolution or fps for detection — far-side ball is ~10px;
  consensus quality is the one thing physics can't recover.

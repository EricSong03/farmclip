# Loop goal — physics-first ball reconstruction (set 2026-07-27)

Loop until ALL of these hold, then stop:

1. **Spec complete**: consensus anchors, velocity-break segmentation (done);
   drag on fast arcs; streak ellipse measurement (direction hint + diameter);
   size-depth hints fused into the fit; net/floor hard anchors where present.
2. **Accurate, measured on BOTH clips (menlo + mikasa)**:
   - holdout median ≤ 3px and p90 ≤ 12px
   - ≥ 85% of held-out anchors covered by an arc
3. **Visually clean**: overlay sheets show no junk arcs (no static-object
   "trajectories", no flat cross-gym lines), verified by inspecting renders.

Progress log (newest first):
- iter 3 (menlo): tight touch-bridging (0.25s/2.5m — the flat cross-gym
  lines were 24m linear bridges, not fits), physics-driven merge pass,
  greedy rescue of over-cut fragments, fragment absorption into arcs.
  menlo: holdout median 2.4px ✓, p90 7.5px ✓, 83% held-out coverage
  (target 85), sheet visually clean except f1511/f377 fit divergence.
  Next: those two arcs, mikasa numbers, then drag + streak/size stages.
- iter 2: stationarity filter added — no effect on menlo (junk anchors
  turned out to be the real ball at ceiling height); flat lines were
  legacy bridging, fixed in iter 3.
- iter 1: core shipped. menlo 55 arcs, holdout median 2.4px / p90 29.2px,
  70% held-out coverage, 41% frame coverage.

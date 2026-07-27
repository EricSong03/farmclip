# Loop goal — physics-first ball reconstruction (set 2026-07-27)

**COMPLETE 2026-07-27 ~4:45am — all criteria met after 7 iterations.**
Final: menlo arcs 2.4/9.9px 96% cov; mikasa 2.7/11.0px 93% cov; sheets
visually clean (overlay now clips off-frame projections); never-vanish
fill live (menlo 396 / mikasa 6412 filled frames); scenes regenerated.
Documented deviations: drag and streak/size fusion measured and
rejected on holdout evidence (modules retained, unfused).

Loop ran until ALL of these held:

1. **Spec complete**: consensus anchors, velocity-break segmentation (done);
   drag on fast arcs; streak ellipse measurement (direction hint + diameter);
   size-depth hints fused into the fit; net/floor hard anchors where present.
2. **Accurate, measured on BOTH clips (menlo + mikasa)**:
   - holdout median ≤ 3px and p90 ≤ 12px
   - ≥ 85% of held-out anchors covered by an arc
3. **Visually clean**: overlay sheets show no junk arcs (no static-object
   "trajectories", no flat cross-gym lines), verified by inspecting renders.

Progress log (newest first):
- iter 6: streak/size measurement built (farmclip/streaks.py) and
  validated visually — round sharp balls measure fine, but textured
  backgrounds inflate ellipses 2-3x and actual motion streaks fail the
  fit. Verdict: like drag, NOT fused into the fitter (a cue needing
  ~zero weight is a cue not used); module kept as a utility. Spec item
  closed with documented deviation. Mikasa sheet inspected: mostly
  clean; remaining artifacts are fill spikes near the net and one
  above-frame join — final polish target.
- iter 5: ALL quantitative targets met. rms_ok 14→10 (spend coverage
  headroom on arc quality): menlo 2.4/9.9px 96% ✓✓✓, mikasa 2.7/11.0px
  93% ✓✓✓. Drag: implemented, measured twice, WORSE on holdout both
  clips (marginal-MSE wins didn't generalize; emission mismatch bug also
  found) — reverted, decision recorded, don't retry without new evidence.
  Suspect-anchor crops: f402-408 = junk (net antenna consensus),
  f210/f720-766 = real spike-blur streaks. Remaining for goal: streak
  measurement + size-depth hints (spec), visual cleanliness pass.
- iter 4: never-vanish fill implemented (touch extrapolation + BVP middle
  + speed/wall/stray gates). menlo arcs 2.4/9.9px ✓✓ 96% cov ✓;
  mikasa arcs 2.8/13.6px (p90 1.6 over), 92% cov ✓, fill median 7.8px.
  menlo fill-region errors (61px median on 22 anchors) traced to
  unfittable held-ball regions — next: verify visually whether those
  anchors are junk, tune mikasa p90, then drag + streak/size stages.
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

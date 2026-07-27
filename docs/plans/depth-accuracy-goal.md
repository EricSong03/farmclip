# Loop goal — 3D depth accuracy (set 2026-07-27, follows physics-first-goal)

2D reprojection is solved (see physics-first-goal.md); depth along the
camera ray is not — it's inferred from arc curvature only, which is
under-determined for short/flat/camera-axis arcs. Convert depth from
inferred to measured.

Loop until ALL hold on BOTH clips, then stop:

1. **Net-crossing error**: when the 2D track passes the net's image band,
   the fitted 3D path crosses the net plane |x| ≤ 0.5m at that moment.
2. **Landings in court**: ≥80% of rally-end floor landings inside the
   court boundary + 1m apron.
3. Reprojection holdout stays at goal (median ≤3px, p90 ≤12px, ≥85%).

Fix ladder (stop climbing when targets hold):
1. Net-plane residual in fit_segment (trigger: 2D track crosses the
   projected net band; residual: x(t_cross) -> 0).
2. Floor-bounce anchors (2D V-reversal near floor -> ray∩floor = exact
   3D point, residual on both adjacent arcs).
3. Rally-chain joint fit (shared touch positions across arcs).
4. Short-arc demotion (<0.4s, unanchored -> depth from chain or no 3D).
5. Periodic recalibration (~10s cadence) for camera drift.
Later (needs ICRN player model, running in parallel): player-contact
anchors at touches.

Rejected already (do not retry without new evidence): ball-size depth
(2-3x ellipse overshoot at 720p), drag (worse holdout, twice).

Progress log (newest first):
- iter 2-4: rally-chain joint fit + net-plane anchors (in-chain, in fills,
  in rally-end extensions) + floor-bounce ray anchors + sweep-aware
  metric. Real crossings: menlo 2.4m -> 0.06m (arc) / 0.54m (fill);
  mikasa 5.5m -> 0.41m (arc) / 0.86m (fill). Bound-parking 13.8% -> 8.5%.
  Reprojection at goal on both. Remaining: fill-crossing tails, landings
  (menlo 4/6, mikasa 76% vs 80%), p90 tails. Per-arc net residual
  attempt (pre-chain) degraded pixels and was reverted — the chain
  formulation is the one that works.
- (pending) iter 1: baseline 3D metrics before any fitter change.

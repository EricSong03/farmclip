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
- (pending) iter 1: baseline 3D metrics before any fitter change.

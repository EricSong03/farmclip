# Ball never vanishes (in play) — grilled 2026-07-27

Goal: continuous ball position for every frame from a rally's first fitted
arc to its floor landing. Between rallies (dead time) the ball may exit.
Scene stays seamless — no provenance flags; debug overlays remain the audit
trail.

## Rally grouping

Adjacent arcs belong to the same rally when BOTH:
- gap ≤ 2.5s (longest realistic mid-rally detection dropout), and
- **joinable**: A extrapolated forward and B backward pass within ≤3m of
  each other somewhere in the gap (physics votes on its own applicability).
Fail either test → rally boundary, no fill.

## Gap fill: extrapolate to touch

For arcs A→B in one rally:
1. Continue A ballistically (same params) forward; continue B backward.
2. Touch time t* = argmin over the gap of |A(t) − B(t)|.
3. Fill [A_end, t*] from A's extrapolation, [t*, B_start] from B's.
   One inferred contact kink; every filled frame obeys gravity.
4. Positional mismatch at t* is closed by distributing the residual
   linearly across the gap (small by the ≤3m joinability gate).

## Rally edges

- End: continue the LAST arc until y=0 (floor) or +1s, whichever first.
- Start: no lead-in extension (no evidence for the serve toss).

## Where

`farmclip/ball3d.py lift()` — replaces the current linear bridge step
(0.25s/2.5m) entirely: touch-extrapolation subsumes it. The teleport
smoother stays as a last-resort guard. Existing arc fitting untouched.

## Acceptance

- physics_eval holdout unchanged (fill adds frames, must not move arcs).
- New metric printed by physics_eval: % of rally-span frames with ball
  (target: 100% by construction — assert it).
- Overlay sheet: filled stretches drawn in a second color for the debug
  render only; visually inspect that inferred touches land near players.

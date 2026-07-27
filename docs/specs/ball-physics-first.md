# Ball tracking, physics-first

Replaces the ML-centric lift (reject_outliers / fill_gaps / per-segment
validation) with physics as the primary reconstruction; the detector's only
job is supplying anchors. Decisions below were grilled 2026-07-27.

## Anchors

- **Consensus detections only**: both VballNet models agree within 12px.
  Single-model and interpolated points are never fit inputs.
- Manual labels (label.html output) are for detector eval/training only, not
  runtime — runtime must work on unlabeled footage.

## Segmentation — velocity breaks, two tiers

- Walk the consensus track; a contact = velocity discontinuity beyond what
  gravity+drag explains between neighboring anchors.
- **Hard break**: large velocity jump (spike, serve, dig). Arc ends, next arc
  starts fresh.
- **Soft break**: small deviation (net graze, fingertip touch). Arc ends, but
  next arc seeds from the previous arc's exit velocity.
- Every stretch between breaks is ONE trajectory fit as a unit.

## Fit model

- 3D state: position + velocity at arc start; gravity 9.81 fixed; quadratic
  air drag on fast arcs (volleyball Cd~0.5, d=21cm, m=270g), pure parabola
  when peak speed is low enough that drag is sub-pixel. No Magnus/float
  modeling — high residual on a float serve is signal, not failure.
- 2D anchors constrain via projection: minimize reprojection error of the 3D
  arc against consensus pixels (camera pose from existing calibration).
- **Depth cues, fused**: gravity curvature is the backbone scale; apparent
  ball size (minor axis of the blob, see streaks) adds weak per-frame depth
  hints weighted by roundness/sharpness; net-plane crossings and floor
  bounces are hard anchors when present.

## Streaks (fast balls)

- At each consensus detection, crop ~60px and ellipse-fit the blob.
  - Round + sharp → clean anchor, minor axis = apparent diameter → depth hint.
  - Elongated → streak: long axis = motion direction hint for the fit; the
    anchor's positional weight drops; minor axis still = diameter.
- A fast stretch is reconstructed from the sharp endpoints (where the ball
  stops being a streak) + streak direction hints in between.

## Output

- Per-frame ball position = evaluated arc, every frame inside a fitted arc
  (including off-screen frames — physics owns those; the detector is never
  asked to guess them).
- Frames outside any accepted arc stay empty. No interpolation without
  physics behind it.

## Acceptance

1. Overlay renders: project fitted arcs onto real footage for full rallies
   into out/debug/, inspect visually (repo convention).
2. Holdout number: refit with a random 20% of consensus anchors removed;
   report px distance from the fit to the held-out anchors. Track this
   number across iterations.

## Sequencing

- Build this first with current detections. ICRN GPU fine-tune happens
  after, and only if anchor density on fast rallies proves too thin.

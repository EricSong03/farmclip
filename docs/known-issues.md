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

## Footage / calibration

- **Court corners rarely visible.** Typical PoV is low, behind the end line — the "click 4 corners" flow in the current stub is unusable on real footage. Superseded by named-keypoint spec; stub UI not yet updated.
- **Distractor lines everywhere.** Multi-sport floors (badminton/basketball lines) and adjacent courts defeat naive line detection. Any automatic detector must be trained/scoped to "the court being played on."
- **Pan/zoom videos.** A single calibration won't hold; needs per-N-frame refresh (planned, not built).

## Stub (index.html)

- "Calibrate" only logs points to console — no pose solve yet.
- Corner overlay coordinates are normalized to the video *element*, not the video *frame* — letterboxing from `object-fit: contain` will skew clicks. Fix when real solving lands.
- Net height fixed at 2.43 m (men's); no toggle yet.

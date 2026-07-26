# Final state — tuning campaign closed (2026-07-26 ~17:00 ET)

USER TARGETS: ball >=90% of trackable time; all players (12 menlo / 4 mikasa)
on court >=80%. Measure: scripts/metrics.py OUTDIR N (reads final scene.json).

## Scoreboard (honest, ghost-free)

| metric              | menlo | mikasa | target | verdict |
|---------------------|-------|--------|--------|---------|
| ball in-play        | 69.6% | 81.5%  | 90     | below   |
| all players >=80%   | 1.3%  | 20.5%  | 80     | blocked |
| typical players     | 7-11  | 2-4    | 12 / 4 |         |

Ball path quality is physics-clean (speeds/heights physical, no teleports).
Player counts are honest: roster cap (6/side, 2/side) makes ghost inflation
impossible — earlier 79-91% numbers were ghost-inflated.

## Why blocked (not tunable further on CPU + this footage)

- Menlo all-12: 720p far side + net occlusion; yolo11s@1280 detects 7-11.
  All 12 simultaneously visible+detected is rare in the raw pixels.
- Mikasa all-4: near players frequently at/off the frame edge (feet not
  visible -> cannot be placed). Camera geometry, not detection.
- Ball 90: remaining gap = non-ballistic phases (serve prep, held) inside
  the in-play denominator + 15% raw detection misses.

## Paths forward (pick per priorities)

1. ICRN H200 fine-tune: volleyball-specific detector (ball+players) on frames
   from these vods -> the single biggest jump for both metrics; then CPU
   inference per game as usual.
2. Metric definition with user: count "players tracked" per rotation window
   instead of per-frame simultaneity; ball vs actual flight-time denominator.
3. Mikasa near-edge players: allow edge-clipped boxes with estimated feet
   (bbox bottom + body-ratio extrapolation) — recovers ~half the missing 4th.

## Housekeeping

- Samples refreshed: menlo-min0 (H.264 clip + scene), mikasa-min2/min11
  (scene.json fresh; clip.mp4 is mp4v — OpenH264 DLL v2.5.0 does not load
  with OpenCV 5's ffmpeg, so H.264 encode unavailable; mp4v may not play in
  browsers. Sidequest abandoned; fix = matching openh264 version or ffmpeg
  binary encode).
- All tuning committed through "feat(ball): dual-model union..." commit.
- Known issues + roadmap in docs/ still current except player metrics above.

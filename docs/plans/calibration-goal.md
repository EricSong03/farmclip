# Loop goal — calibration fix (set 2026-07-28, grilled)

Overlay videos (out/*/debug/overlay.mp4) show the wireframe off reality:
mikasa net ~70px high + court rotated/wide; menlo net low + court rotated.
Ball tracking is pixel-true; every 3D coordinate inherits the pose bias.

Decisions (grilled):
- Fix method: AUTO, harder — seed from current calib, add measured net
  band/posts/attack lines, line-ICP refinement, try multiple reference
  frames. Ask user for manual clicks ONLY if auto plateaus.
- Net height: free-parameter SUPPORT built for future venues, but BOTH
  current videos are strictly FIVB regulation (user-confirmed) — fit them
  with net height PINNED at 2.43m. The mikasa 70px net offset is
  therefore pure pose error. Court dims stay FIVB unless line residuals
  prove otherwise (then report evidence, ask before changing).
- Acceptance (BOTH clips): median projected-vs-detected line distance
  <=5px (net band <=8px) at >=5 timestamps spread across the video, AND
  visual overlay check each iteration (wireframe on the real lines).
- Downstream in-loop: ball3d + scenes + overlay videos regenerate on
  accept. Player re-run (fine-tuned yolo11s-vb.pt) launches as a
  background job at loop END — floor rays need the new pose anyway.

Progress log (newest first):
- iter 1: scorer built (scripts/calib_eval.py: projected model lines vs
  detected segments, clipped-mean + miss penalty) and RANSAC refiner
  (scripts/calib_refine.py, scorer as judge, focal anchored). Baselines:
  menlo court 19.3/net 14.5px, mikasa court 14.3/net 48.8px. Learned the
  hard way: naive ICP diverges (distractor lines), focal must be bounded
  to the ORIGINAL value (compounded 1459->2667 across reruns), median
  score is cheatable by dropping lines. Menlo currently on a fresh
  auto-calib + refine that is VISUALLY WORSE (squashed court) — the old
  pose is gone (also wrong, nothing precious). Next: multi-hypothesis
  refinement (refine EVERY calibrate() candidate frame, not just the
  inlier-count winner), measured net band + posts as high-weight
  constraints, then mikasa. calib_orig.json snapshots now prevent
  focal-anchor loss.

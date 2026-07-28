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
- (pending) iter 1: build line-alignment scorer, baseline both clips.

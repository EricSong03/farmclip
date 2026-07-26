# Spec: Court + Net Calibration from Video

## Goal

Given gameplay footage (usually stationary camera, low angle behind the end line), recover the camera pose so the 3D court model (lines + net) aligns with the video.

## Constraints observed from real footage (examples/)

- Camera low, behind end line: near corners and near end line out of frame.
- Multi-sport floor: badminton/basketball lines cross volleyball lines.
- Cluttered background: adjacent courts with their own nets/lines, banners, dome netting.
- Some videos pan/zoom → pose must be refreshable per-frame or every N frames.

## Approach

**Pose from known 3D points, not homography-from-corners.**

Court model (FIVB, meters, Y-up, origin at court center under net):
- Court 18 × 9, attack lines ±3 from center, line width 0.05.
- Net height 2.43 (men) / 2.24 (women) — must be selectable.
- Net width 9.5, posts 0.5–1.0 outside sidelines, antennas above sideline intersections.

Named keypoint set (each with fixed 3D coords):
- Line intersections: sideline × attack line (×4 visible-ish), sideline × center line (×2), far corners (×2, when visible).
- Off-floor: net post base/top (×2 each), antenna tips (×2), net top band × sideline plane.

Solve: `cv2.solvePnP` (or JS equivalent) with whichever ≥4 non-degenerate keypoints are available. Off-floor points break the coplanarity degeneracy — prefer including at least one.

## Phases (revised 2026-07-26: manual v1 dropped, straight to auto)

1. **Automatic:** existing open-source volleyball court keypoint/segmentation models (U-Net court detector, etc.) predict the named keypoints → solvePnP. Validation is reprojection-based (projected court lines must hug the video's lines in overlay debug frames) — no hand labeling. Fine-tune/train (synthetic renders are the fallback data source) only if off-the-shelf fails.
2. **Refresh:** re-run detection every N frames + temporal pose smoothing. Stationary video converges to a fixed pose; pan/zoom tracks.

## Non-goals (for now)

- Player/ball detection and tracking.
- Multi-camera fusion.
- Real-time performance.

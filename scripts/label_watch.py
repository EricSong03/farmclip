"""Watch web-image labels and render solved-court overlays for the clicker.

Run alongside calib.html: python -m uv run python scripts/label_watch.py
For each out/webimgs/labels/<stem>.json (>=5 pts) whose overlay is missing or
stale, solve the camera from the clicks and write the wireframe overlay to
out/webimgs/overlays/<stem>.jpg with the reprojection error stamped on it.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.calibrate import draw_overlay, solve_web
from farmclip.court import KEYPOINTS


def floor_spread_ok(ann):
    """False when floor clicks are too few or near-collinear (pose underdetermined)."""
    pts = np.array([v for k, v in ann.items()
                    if k in KEYPOINTS and KEYPOINTS[k][1] == 0], float)
    if len(pts) < 6:
        return False
    _, s, _ = np.linalg.svd(pts - pts.mean(0))
    return s[1] / s[0] > 0.08


def worst_click(ann, w, h, err, trigger=15.0, ratio=0.3):
    """Name the single click that is wrecking the solve, if there is one.

    Least squares spreads one bad point's error across every other point, so a
    single mis-click makes the WHOLE overlay look wrong rather than one corner.
    Leave-one-out over the floor points finds it: on a real label set, dropping
    corner_far_right took the fit from 340.9px to 5.8px.

    Only runs when the fit is already bad, since it costs one solve per point.
    """
    from farmclip.calibrate import lm_polish, solve
    floor = {k: v for k, v in ann.items()
             if k in KEYPOINTS and KEYPOINTS[k][1] == 0}
    if err < trigger or len(floor) < 6:
        return None
    best = None
    for k in floor:
        sub = {x: v for x, v in floor.items() if x != k}
        try:
            e = lm_polish(solve(sub, w, h), sub)["err"]
        except Exception:
            continue
        if best is None or e < best[1]:
            best = (k, e)
    return best if best and best[1] < err * ratio else None


def depth_span(ann):
    """Metres of COURT LENGTH the clicks cover (world X). The other degeneracy.

    Points bunched at one end leave focal length and camera distance trading
    off almost exactly, so the solve slides to the focal clamp and returns a
    confident-looking but wrong pose. Measured on a real label set: points at
    x=0 and x=+3 only -> 78.8px err with f pinned at the 3.5*w bound; adding a
    single far-end point (x=-9) -> 8.9px and f free at 1947. Image-plane
    spread (floor_spread_ok) does NOT catch this — those clicks looked well
    spread on screen.
    """
    xs = [KEYPOINTS[k][0] for k in ann if k in KEYPOINTS]
    return (max(xs) - min(xs)) if xs else 0.0

WEB = Path(__file__).parent.parent / "out/webimgs"
OVL = WEB / "overlays"
OVL.mkdir(parents=True, exist_ok=True)

print(f"watching {WEB / 'labels'} ... ctrl-c to stop")
while True:
    for lp in sorted((WEB / "labels").glob("*.json")):
        dst = OVL / (lp.stem + ".jpg")
        if dst.exists() and dst.stat().st_mtime >= lp.stat().st_mtime:
            continue
        img_path = next((p for e in (".jpg", ".jpeg", ".png")
                         if (p := WEB / (lp.stem + e)).exists()), None)
        if img_path is None:
            continue
        try:
            ann = json.loads(lp.read_text())
        except (json.JSONDecodeError, OSError):  # mid-write
            continue
        ann = {k: v for k, v in ann.items() if not k.startswith("post_")}
        frame = cv2.imread(str(img_path))
        if frame is None or len(ann) < 5:
            continue
        h, w = frame.shape[:2]
        try:
            calib, net_h = solve_web(ann, w, h)
            out = draw_overlay(frame, calib, ann, net_h=net_h)
            e = calib["err"]
            color = (0, 200, 0) if e < 15 else (0, 165, 255) if e < 40 else (0, 0, 255)
            verdict = "good" if e < 15 else "check worst points" if e < 40 else "BAD - relabel"
            nh = f" | net ~{net_h:.2f}m" if net_h else ""
            bad = worst_click(ann, w, h, e)
            span = depth_span(ann)
            if bad:  # one mis-click poisons every point — name it
                color = (0, 0, 255)
                verdict = f"BAD POINT: {bad[0]} (without it {bad[1]:.1f}px)"
            elif span < 8.0:  # under ~half the court's length: focal/depth degenerate
                color = (0, 165, 255)
                verdict = (f"SHALLOW ({span:.0f}m of court) - click the FAR end "
                           f"(far corners / far attack line)")
            elif not floor_spread_ok(ann):
                color = (0, 165, 255)
                verdict = "UNDERDETERMINED - click more floor pts (both sides!)"
            cv2.putText(out, f"floor err {e:.1f}px - {verdict}{nh}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        except Exception as ex:
            out = frame.copy()
            cv2.putText(out, "solve failed - need >=5 good pts", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            print(f"{lp.stem}: solve failed ({ex})")
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # label mtimes can sit in the future (browser-written) — pin the
        # overlay's mtime to the label's so the staleness check terminates
        os.utime(dst, (lp.stat().st_mtime, lp.stat().st_mtime))
        print(f"{lp.stem}: overlay written")
    time.sleep(1)

"""Watch web-image labels and render solved-court overlays for the clicker.

Run alongside calib.html: python -m uv run python scripts/label_watch.py
For each data/pool/labels/<stem>.json (>=5 pts) whose overlay is missing or
stale, solve the camera from the clicks and write the wireframe overlay to
data/pool/overlays/<stem>.jpg with the reprojection error stamped on it.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.calibrate import draw_overlay, solve_labeled
from farmclip.court import KEYPOINTS, depth_span, width_span


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


ROOT = Path(__file__).parent.parent
# Both pools: data/pool is training, data/test is the held-out benchmark.
# The clicker can write to either, so the watcher has to render overlays for
# both or the test set is labelled blind.
POOLS = [d for d in (ROOT / "data/pool", ROOT / "data/test") if d.exists()]
for _p in POOLS:
    (_p / "overlays").mkdir(parents=True, exist_ok=True)

def render(lp, img_path, dst):
    """Solve one label set and write its stamped overlay. Shared by the image
    pools and the per-run annotations -- a run is labelled through the same UI
    and was previously getting no overlay at all, which meant labelling the
    highest-yield data was the one case done blind."""
    if dst.exists() and dst.stat().st_mtime >= lp.stat().st_mtime:
        return
    try:
        ann = json.loads(lp.read_text())
    except (json.JSONDecodeError, OSError):  # mid-write
        return
    ann = {k: v for k, v in ann.items() if not k.startswith("post_")}
    frame = cv2.imread(str(img_path))
    if frame is None or len(ann) < 5:
        return
    h, w = frame.shape[:2]
    try:
        # solve_labeled, NOT solve_web: solve_web re-votes L/R pairs and
        # tries the mirrored floor assignment, i.e. it overrides the
        # names the human just asserted. Here the clicks are ground
        # truth and the pose is whatever they imply -- including an
        # impossible one, which gets reported rather than repaired.
        calib = solve_labeled(ann, w, h)
        net_h = calib.get("net_h_est")
        out = draw_overlay(frame, calib, ann, net_h=net_h)
        e = calib["err"]
        color = (0, 200, 0) if e < 15 else (0, 165, 255) if e < 40 else (0, 0, 255)
        verdict = "good" if e < 15 else "check worst points" if e < 40 else "BAD - relabel"
        nh = ((f" | net ~{net_h:.2f}m" if net_h else "")
              + f" | cam {calib['cam_height']}m up")
        bad = worst_click(ann, w, h, e)
        span = depth_span(ann)
        if calib["cam_below_floor"]:
            # clicks satisfiable only by a camera under the court:
            # near/far or L/R is flipped. f / g in the clicker.
            color = (0, 0, 255)
            verdict = "IMPOSSIBLE POSE (camera under floor) - press f or g"
        elif bad:
            color = (0, 0, 255)
            verdict = f"BAD POINT: {bad[0]} (without it {bad[1]:.1f}px)"
        elif width_span(ann) < 4.0:
            # every click in one vertical plane (one sideline, or the net
            # and a sideline on the same side). The pose is not
            # determined, and the residual will look GREAT anyway -- the
            # three worst frames in the held-out set scored 0.4-1.9px.
            color = (0, 0, 255)
            verdict = ("ONE SIDELINE ONLY - pose undetermined despite the "
                       "low error. Click the OTHER sideline.")
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
        print(f"{lp.parent.name if lp.stem == 'annotations' else lp.stem}: solve failed ({ex})")
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    # label mtimes can sit in the future (browser-written) — pin the
    # overlay's mtime to the label's so the staleness check terminates
    os.utime(dst, (lp.stat().st_mtime, lp.stat().st_mtime))
    name = lp.parent.name if lp.stem == "annotations" else lp.stem
    print(f"{name}: overlay written")


# Per-run annotations: data/runs/<name>/annotations.json against ref_frame.jpg.
RUNS_DIR = ROOT / "data/runs"

print(f"watching {', '.join(str(p / 'labels') for p in POOLS)} + {RUNS_DIR} ... ctrl-c to stop")
while True:
    for WEB in POOLS:
        OVL = WEB / "overlays"
        for lp in sorted((WEB / "labels").glob("*.json")) if (WEB / "labels").exists() else []:
            img_path = next((p for e in (".jpg", ".jpeg", ".png")
                             if (p := WEB / (lp.stem + e)).exists()), None)
            if img_path is not None:
                render(lp, img_path, OVL / (lp.stem + ".jpg"))
    for rd in sorted(RUNS_DIR.glob("*/")) if RUNS_DIR.exists() else []:
        lp, ref = rd / "annotations.json", rd / "ref_frame.jpg"
        if lp.exists() and ref.exists():
            render(lp, ref, rd / "overlay.jpg")
    time.sleep(1)

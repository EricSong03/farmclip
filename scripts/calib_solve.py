"""Solve calib from manual annotations, trying the left/right-swapped naming too.

Which side is 'left' depends on where the camera stands; a full swap is the
most common annotation mistake and produces exactly one signature: corners
100px+ off while net points are fine. Solve both interpretations, best wins.
Usage: python scripts/calib_solve.py <outdir>
"""
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.calibrate import solve, draw_overlay

outdir = Path(sys.argv[1])
ann = json.loads((outdir / "annotations.json").read_text())
frame = cv2.imread(str(outdir / "debug" / "ref_frame.jpg"))
h, w = frame.shape[:2]


NET_GROUP = ("antenna_tip", "net_top", "post_top", "post_base")


def swap_lr(a, only=None):
    def sw(k):
        if only is not None and not k.startswith(only):
            return k
        if k.endswith("_left"):
            return k[:-5] + "_right"
        if k.endswith("_right"):
            return k[:-6] + "_left"
        return k
    return {sw(k): v for k, v in a.items()}


def swap_group(a, group):
    out = dict(a)
    for pre in group:
        out = swap_lr(out, only=pre)
    return out


cands = []
for label, a in [("as-clicked", ann),
                 ("lr-swapped", swap_lr(ann)),
                 ("net-group-swapped", swap_group(ann, NET_GROUP)),
                 ("floor-group-swapped", swap_lr(swap_group(ann, NET_GROUP)))]:
    try:
        c = solve(a, w, h)
        cands.append((c["err"], label, c, a))
        print(f"{label}: err {c['err']:.1f}px, f {c['f']:.0f}")
    except Exception as e:
        print(f"{label}: solve failed ({e})")

err, label, calib, used = min(cands, key=lambda x: x[0])

# continuous LM polish over (rvec, tvec, f): solve()'s focal grid is coarse
# (0.05*w steps) and solvePnP alone leaves tens of px on the table
import numpy as np
from scipy.optimize import least_squares
from farmclip.court import KEYPOINTS

names = [k for k in used if k in KEYPOINTS]
obj = np.array([KEYPOINTS[k] for k in names], float)
img = np.array([used[k] for k in names], float)


def resid(p):
    rvec, tvec, f = p[:3], p[3:6], p[6]
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    return (proj.reshape(-1, 2) - img).ravel()


p0 = np.concatenate([calib["rvec"], calib["tvec"], [calib["f"]]])
sol = least_squares(resid, p0, method="lm", max_nfev=5000)
new_err = float(np.sqrt(np.mean(np.sum(resid(sol.x).reshape(-1, 2) ** 2, axis=1))))
if new_err < err:
    calib = dict(calib, rvec=sol.x[:3].tolist(), tvec=sol.x[3:6].tolist(),
                 f=float(sol.x[6]), err=new_err)
    d = np.linalg.norm(resid(sol.x).reshape(-1, 2), axis=1)
    calib["per_point_err"] = {k: float(e) for k, e in zip(names, d)}
    err = new_err
    print(f"LM polish: err -> {new_err:.1f}px, f {calib['f']:.0f}")
(outdir / "calib.json").write_text(json.dumps(calib, indent=1))
cv2.imwrite(str(outdir / "debug" / "calib_overlay.jpg"),
            draw_overlay(frame, calib, used))
print(f"WINNER: {label} err {err:.1f}px -> {outdir / 'calib.json'} "
      f"(overlay: debug/calib_overlay.jpg)")
if label == "lr-swapped":
    (outdir / "annotations.json").write_text(json.dumps(used, indent=1))
    print("annotations.json rewritten with corrected names")

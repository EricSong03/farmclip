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


def swap_lr(a):
    def sw(k):
        if k.endswith("_left"):
            return k[:-5] + "_right"
        if k.endswith("_right"):
            return k[:-6] + "_left"
        return k
    return {sw(k): v for k, v in a.items()}


cands = []
for label, a in [("as-clicked", ann), ("lr-swapped", swap_lr(ann))]:
    try:
        c = solve(a, w, h)
        cands.append((c["err"], label, c, a))
        print(f"{label}: err {c['err']:.1f}px, f {c['f']:.0f}")
    except Exception as e:
        print(f"{label}: solve failed ({e})")

err, label, calib, used = min(cands, key=lambda x: x[0])
(outdir / "calib.json").write_text(json.dumps(calib, indent=1))
cv2.imwrite(str(outdir / "debug" / "calib_overlay.jpg"),
            draw_overlay(frame, calib, used))
print(f"WINNER: {label} err {err:.1f}px -> {outdir / 'calib.json'} "
      f"(overlay: debug/calib_overlay.jpg)")
if label == "lr-swapped":
    (outdir / "annotations.json").write_text(json.dumps(used, indent=1))
    print("annotations.json rewritten with corrected names")

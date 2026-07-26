"""Hypothesis-search calibration on a frame's detected segments."""
import json
import sys

import cv2

from farmclip.calibrate import draw_overlay
from farmclip.hypothesis import search

tag = sys.argv[1] if len(sys.argv) > 1 else "0600"
frame_path = sys.argv[2] if len(sys.argv) > 2 else f"out/debug/frame_{tag}.jpg"
segs = json.load(open(f"out/segments_{tag}.json"))

# anchors: attack_near = longest low horizontal; net_band = seg through (286,273) for 0600
# generalize: pick per-frame via simple rules
def slope(s):
    return abs(s[3] - s[1]) / max(abs(s[2] - s[0]), 1)

low_horiz = [s for s in segs if slope(s) < 0.15 and (s[1] + s[3]) / 2 > 430]
attack_near = max(low_horiz, key=lambda s: abs(s[2] - s[0]))
band_cands = [s for s in segs if slope(s) < 0.1 and 200 < (s[1] + s[3]) / 2 < 300
              and abs(s[2] - s[0]) > 600]
net_band = min(band_cands, key=lambda s: abs((s[1] + s[3]) / 2 - 265)) if band_cands else None
print("attack_near", [round(v) for v in attack_near])
print("net_band", [round(v) for v in net_band] if net_band else None)

calib, info = search(segs, attack_near, net_band, 1280, 720, verbose=True)
if calib is None:
    print("NO HYPOTHESIS PASSED")
    sys.exit(1)
print("BEST:", {k: (round(v, 1) if isinstance(v, float) else v)
                for k, v in info.items() if k not in ("segL", "segR", "segF")})

# tighten with scipy least-squares point-to-line polish
from farmclip.court import HL, HW, ATTACK, NET_H
from farmclip.hypothesis import FAR_LINES
from farmclip.refine_lsq import polish

floor_pairs = [
    (attack_near, ((ATTACK, 0, -HW), (ATTACK, 0, +HW))),
    (info["segL"], ((-HL, 0, +HW), (HL, 0, +HW))),
    (info["segR"], ((-HL, 0, -HW), (HL, 0, -HW))),
    (info["segF"], FAR_LINES[info["far"]]),
]
# measured post pad bases on frame 1770 (image-left = world +Z)
post_px = [((165, 487), +1), ((1230, 447), -1)] if tag == "f1770" else ()
calib = polish(calib, floor_pairs, band_seg=net_band, post_px=post_px)
print("polished err", round(calib["err"], 1), "f", round(calib["f"]),
      "net_h", round(calib["net_h"], 2), "post_hw", round(calib["post_hw"], 2))
json.dump(calib, open("out/calib.json", "w"), indent=1)
frame = cv2.imread(frame_path)
cv2.imwrite(f"out/debug/calib_hyp_{tag}.jpg", draw_overlay(frame, calib))
print("overlay: out/debug/calib_hyp_" + tag + ".jpg")

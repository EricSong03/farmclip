"""Multi-hypothesis calibration: refine EVERY auto-calib candidate, best wins.

The auto solver keeps only its inlier-count winner, which can be confidently
wrong (menlo: 7 inliers, squashed court). Here every candidate frame's
hypothesis gets the full RANSAC refinement and the calib_eval scorer picks
the survivor. Usage: python scripts/calib_search.py <outdir> <video>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip import video as fvideo
from farmclip.lines import detect_segments
from farmclip.hypothesis import search, search_headon, FAR_LINES
from farmclip.refine_lsq import polish
from farmclip.court import HL, HW, ATTACK
from calib_refine import build_seg_cache, refine_ransac, score

outdir, vid = Path(sys.argv[1]), sys.argv[2]

# --- gather ALL candidates (mirror of cli.calibrate's scan, keeping each) ---
info_v = fvideo.video_info(vid)
w, h = info_v["width"], info_v["height"]
step = max(1, int(info_v["frames"] // 40))
cands = []
for i, t, frame in fvideo.read_frames(str(vid), step=step):
    segs = detect_segments(frame)

    def slope(s):
        return abs(s[3] - s[1]) / max(abs(s[2] - s[0]), 1)

    def ymid(s):
        return (s[1] + s[3]) / 2

    low = [s for s in segs if slope(s) < 0.15 and ymid(s) > h * 0.6]
    band = [s for s in segs if slope(s) < 0.1 and h * 0.25 < ymid(s) < h * 0.45
            and abs(s[2] - s[0]) > w * 0.45]
    if not band:
        continue
    paired = [s for s in band
              if any(20 < ymid(o) - ymid(s) < 80 for o in band if o is not s)]
    pool = paired or band
    net_band = min(pool, key=ymid) if paired else max(band, key=lambda s: abs(s[2] - s[0]))
    calib = None
    if low:
        attack_near = max(low, key=lambda s: abs(s[2] - s[0]))
        calib, info = search(segs, attack_near, net_band, w, h)
        if calib is not None:
            calib = polish(calib, [
                (attack_near, ((ATTACK, 0, -HW), (ATTACK, 0, +HW))),
                (info["segL"], ((-HL, 0, +HW), (HL, 0, +HW))),
                (info["segR"], ((-HL, 0, -HW), (HL, 0, -HW))),
                (info["segF"], FAR_LINES[info["far"]]),
            ], band_seg=net_band)
    if calib is None:
        calib, info = search_headon(segs, net_band, w, h)
    if calib is not None:
        calib["ref_frame"] = i
        cands.append(calib)
print(f"{len(cands)} candidates")

# --- refine every candidate; scorer picks the survivor ---
seg_cache = build_seg_cache(vid)
cur = outdir / "calib.json"
if cur.exists():
    c = json.loads(cur.read_text())
    cands.append(c)  # incumbent competes too
best = None
for j, cand in enumerate(cands):
    ref, s = refine_ransac(cand, seg_cache, f_anchor=cand["f"], trials=40, seed=j)
    print(f"cand {j} (ref f{cand.get('ref_frame', '?')}): "
          f"court {s[1]:.1f}px net {s[2]:.1f}px")
    if best is None or s[0] < best[1][0]:
        best = (ref, s)

ref, s = best
if cur.exists():
    (outdir / "calib_old.json").write_text(cur.read_text())
cur.write_text(json.dumps(
    {k: v for k, v in ref.items() if not str(k).startswith("_")}, indent=1))
print(f"WINNER: court {s[1]:.1f}px net {s[2]:.1f}px -> {cur}")

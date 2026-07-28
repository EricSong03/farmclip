"""Calibration alignment score: projected court/net lines vs detected pixels.

Usage: python scripts/calib_eval.py <outdir> <video> [calib.json]
Scores median perpendicular px distance from detected segments to their
matched projected model line, at 6 timestamps. Goal: <=5px court, <=8px net.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.lines import detect_segments
from farmclip.ball3d import _project
from farmclip.court import HL, HW, ATTACK, NET_H

MODEL = {
    "side1": [(-HL, 0, -HW), (HL, 0, -HW)], "side2": [(-HL, 0, HW), (HL, 0, HW)],
    "end1": [(-HL, 0, -HW), (-HL, 0, HW)], "end2": [(HL, 0, -HW), (HL, 0, HW)],
    "center": [(0, 0, -HW), (0, 0, HW)],
    "att1": [(-ATTACK, 0, -HW), (-ATTACK, 0, HW)],
    "att2": [(ATTACK, 0, -HW), (ATTACK, 0, HW)],
    "net": [(0, NET_H, -HW), (0, NET_H, HW)],
}


def seg_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) % np.pi


def match_segs(segs, calib, perp_max=60):
    """Associate pre-detected segments to model lines under the current pose.
    Returns {name: (err_px, seg)}."""
    per_line = {}
    for name, (a, b) in MODEL.items():
        uv = _project(calib, [a, b])
        p, q = uv[0], uv[1]
        L = np.linalg.norm(q - p)
        if L < 40:
            continue
        d_unit = (q - p) / L
        n_unit = np.array([-d_unit[1], d_unit[0]])
        ang = seg_angle(p[0], p[1], q[0], q[1])
        best = None
        for s in segs:
            sl = np.hypot(s[2] - s[0], s[3] - s[1])
            if sl < L * 0.25:
                continue
            if abs((seg_angle(*s[:4]) - ang + np.pi / 2) % np.pi - np.pi / 2) > 0.12:
                continue
            m = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
            along = np.dot(m - p, d_unit)
            if along < -0.2 * L or along > 1.2 * L:
                continue
            perp = abs(np.dot(m - p, n_unit))
            if perp > perp_max:
                continue
            e = np.mean([abs(np.dot(np.array(s[i:i + 2]) - p, n_unit)) for i in (0, 2)])
            if best is None or e < best[0]:
                best = (e, s)
        if best is not None:
            per_line[name] = best
    return per_line


def match_frame(img, calib, perp_max=60):
    return match_segs(detect_segments(img), calib, perp_max)


def score_frame(img, calib):
    return {k: v[0] for k, v in match_frame(img, calib).items()}


def evaluate(video, calib, n_stamps=6):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    court_e, net_e, per = [], [], {}
    for fr in np.linspace(0, n - 1, n_stamps).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, img = cap.read()
        if not ok:
            continue
        s = score_frame(img, calib)
        for k, v in s.items():
            per.setdefault(k, []).append(v)
            (net_e if k == "net" else court_e).append(v)
    cap.release()
    return court_e, net_e, per


if __name__ == "__main__":
    outdir, video = Path(sys.argv[1]), sys.argv[2]
    cpath = Path(sys.argv[3]) if len(sys.argv) > 3 else outdir / "calib.json"
    calib = json.loads(cpath.read_text())
    court_e, net_e, per = evaluate(video, calib)
    for k, v in sorted(per.items()):
        print(f"  {k}: median {np.median(v):.1f}px over {len(v)} stamps")
    print(f"COURT median {np.median(court_e):.1f}px (goal <=5) | "
          f"NET median {np.median(net_e) if net_e else float('nan'):.1f}px (goal <=8) | "
          f"{len(court_e)} court matches")

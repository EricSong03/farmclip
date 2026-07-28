"""Debug overlay video: court lines + net + reconstructed ball on real footage.

Usage: python scripts/overlay_video.py <outdir> <video> [start_s end_s]
e.g.   python scripts/overlay_video.py out out/clip.mp4
       python scripts/overlay_video.py out/mikasa out/mikasa/clip.mp4 60 120
Writes <outdir>/debug/overlay.mp4 (H.264).
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.ball3d import _project
from farmclip.court import HL, HW, ATTACK, NET_H

outdir, video = Path(sys.argv[1]), sys.argv[2]
calib = json.loads((outdir / "calib.json").read_text())
ball = {int(k): v for k, v in json.loads((outdir / "ball3d.json").read_text()).items()}

COURT = [  # 3D line segments, floor y=0
    [(-HL, 0, -HW), (HL, 0, -HW)], [(-HL, 0, HW), (HL, 0, HW)],       # sidelines
    [(-HL, 0, -HW), (-HL, 0, HW)], [(HL, 0, -HW), (HL, 0, HW)],       # endlines
    [(0, 0, -HW), (0, 0, HW)],                                        # center
    [(-ATTACK, 0, -HW), (-ATTACK, 0, HW)], [(ATTACK, 0, -HW), (ATTACK, 0, HW)],
]
NET = [
    [(0, NET_H, -HW), (0, NET_H, HW)],            # net top
    [(0, NET_H - 1.0, -HW), (0, NET_H - 1.0, HW)],  # net bottom edge
    [(0, 0, -HW), (0, NET_H, -HW)], [(0, 0, HW), (0, NET_H, HW)],  # posts
]


def draw_line3d(img, a, b, color, t=2):
    uv = _project(calib, [a, b])
    h, w = img.shape[:2]
    if all(-2 * w < p[0] < 3 * w and -2 * h < p[1] < 3 * h for p in uv):
        cv2.line(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), color, t)


cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w, h = int(cap.get(3)), int(cap.get(4))
f0 = int(float(sys.argv[3]) * fps) if len(sys.argv) > 3 else 0
f1 = int(float(sys.argv[4]) * fps) if len(sys.argv) > 4 else n
cap.set(cv2.CAP_PROP_POS_FRAMES, f0)

tmp = outdir / "debug" / "_overlay_tmp.mp4"
dest = outdir / "debug" / "overlay.mp4"
tmp.parent.mkdir(exist_ok=True)
vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

for f in range(f0, f1):
    ok, img = cap.read()
    if not ok:
        break
    for seg in COURT:
        draw_line3d(img, *seg, (80, 255, 80))
    for seg in NET:
        draw_line3d(img, *seg, (255, 160, 60))
    trail = [ball[g] for g in range(f - 20, f + 1) if g in ball]
    if trail:
        uv = _project(calib, trail)
        for p, q in zip(uv, uv[1:]):
            if max(abs(p - q)) < w:
                cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), (0, 220, 255), 2)
    if f in ball:
        p = _project(calib, [ball[f]])[0].astype(int)
        if 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(img, tuple(p), 10, (0, 0, 255), 3)
    vw.write(img)
cap.release()
vw.release()

ff = Path(sys.prefix) / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries"
ffexe = next(ff.glob("ffmpeg-*"), None)
if ffexe:
    subprocess.run([str(ffexe), "-y", "-v", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                    "-an", str(dest)], check=True)
    tmp.unlink()
else:
    tmp.rename(dest)  # mp4v fallback: plays in VLC, not browsers
print(f"wrote {dest} ({f1 - f0} frames)")

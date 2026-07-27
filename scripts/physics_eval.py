"""Physics-first ball eval: overlay contact sheet + anchor-holdout px error.

Usage: python scripts/physics_eval.py <outdir> <video> <fps> <name>
e.g.   python scripts/physics_eval.py out out/clip.mp4 30 menlo
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.ball3d import consensus, reject_outliers, lift, _project, G

outdir, video, fps, name = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4]
calib = json.loads((outdir / "calib.json").read_text())
rng = np.random.default_rng(0)

cols = ["Frame", "X", "Y", "Visibility"]
dfs = [pd.read_csv(list((outdir / f"ball{i}").glob("*/ball.csv"))[0])[cols]
       .to_numpy(float) for i in (0, 1)]
track = reject_outliers(consensus(*dfs))
anchors = track[track[:, 3] > 0]
print(f"{name}: {len(anchors)} consensus anchors / {len(track)} frames")

# --- full fit ---
ball3d, n_seg, n_ok, filled = lift(track, calib, fps, segmenter="velocity")
print(f"full fit: {n_ok} arcs, {len(ball3d)} 3D frames "
      f"({len(ball3d) / len(track) * 100:.0f}% coverage, "
      f"{len(filled)} physics-filled)")

# --- holdout: drop 20% of anchors, refit, measure px error at held-out ones ---
hold = rng.random(len(track)) < 0.20
t2 = track.copy()
t2[hold, 3] = 0
fit2, _, _, filled2 = lift(t2, calib, fps, segmenter="velocity")
held = track[hold & (track[:, 3] > 0)]
arc_e, fill_e = [], []
for fr, x, y, _ in held:
    if int(fr) in fit2:
        uv = _project(calib, [fit2[int(fr)]])[0]
        (fill_e if int(fr) in filled2 else arc_e).append(
            np.hypot(uv[0] - x, uv[1] - y))
n_cov = len(arc_e) + len(fill_e)
if arc_e:
    print(f"HOLDOUT arcs: median {np.median(arc_e):.1f}px, "
          f"p90 {np.percentile(arc_e, 90):.1f}px ({len(arc_e)} anchors)")
if fill_e:
    print(f"HOLDOUT fill: median {np.median(fill_e):.1f}px, "
          f"p90 {np.percentile(fill_e, 90):.1f}px ({len(fill_e)} anchors)")
print(f"HOLDOUT coverage: {n_cov / max(len(held), 1) * 100:.0f}% of {len(held)}")

# --- overlay contact sheet: mid-frame of up to 12 arcs, trail = fitted path ---
frames_fit = sorted(ball3d)
arcs = []
s = frames_fit[0]
for a, b in zip(frames_fit, frames_fit[1:]):
    if b - a > 1:
        arcs.append((s, a)); s = b
arcs.append((s, frames_fit[-1]))
picks = arcs[:: max(1, len(arcs) // 12)][:12]

cap = cv2.VideoCapture(video)
tiles = []
for f0, f1 in picks:
    mid = (f0 + f1) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, img = cap.read()
    if not ok:
        continue
    path = _project(calib, [ball3d[f] for f in range(f0, f1 + 1)])
    for p, q in zip(path, path[1:]):
        cv2.line(img, tuple(p.astype(int)), tuple(q.astype(int)), (255, 200, 0), 2)
    for fr, x, y, _ in anchors[(anchors[:, 0] >= f0) & (anchors[:, 0] <= f1)]:
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.putText(img, f"arc f{f0}-f{f1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    tiles.append(cv2.resize(img, (640, 360)))
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
dest = outdir.parent / "out" / "debug" if outdir.name != "out" else outdir / "debug"
dest = outdir / "debug"
dest.mkdir(exist_ok=True)
cv2.imwrite(str(dest / f"physics_{name}.png"), sheet)
print(f"overlay sheet: {dest / f'physics_{name}.png'} ({len(picks)} arcs shown)")

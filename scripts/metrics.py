"""Accuracy metrics vs targets: ball >=90% of in-play frames, 4+ players >=80%.
Usage: metrics.py OUTDIR [n_players_expected]"""
import glob
import json
import sys

import numpy as np
import pandas as pd

out = sys.argv[1] if len(sys.argv) > 1 else "out/mikasa"
need = int(sys.argv[2]) if len(sys.argv) > 2 else 4

csv = (glob.glob(f"{out}/ball0/*/ball.csv") + glob.glob(f"{out}/ball/*/ball.csv")
       + glob.glob(f"{out}/ball_vballnet/*/ball.csv"))
df = pd.read_csv(csv[0])
vis = df[df.Visibility > 0].Frame.to_numpy()
total = len(df)

# rally spans: gaps > 2s (at clip fps) split play; in-play = union of spans
fps = 60 if total > 30000 else 30
spans, s = [], vis[0] if len(vis) else 0
for a, b in zip(vis, vis[1:]):
    if b - a > 2 * fps:
        spans.append((s, a))
        s = b
if len(vis):
    spans.append((s, vis[-1]))
in_play = sum(b - a + 1 for a, b in spans)
covered = sum(((vis >= a) & (vis <= b)).sum() for a, b in spans)
ball_pct = 100 * covered / max(in_play, 1)
print(f"ball: {len(vis)}/{total} raw ({100*len(vis)//total}%), "
      f"in-play coverage {ball_pct:.1f}% over {len(spans)} rallies "
      f"({in_play/fps:.0f}s in play)  target 90 {'PASS' if ball_pct >= 90 else 'FAIL'}")

try:
    scene = json.load(open(f"{out}/scene.json"))
    counts = {}
    for f in scene["frames"]:
        counts[round(f["t"] * fps)] = len(f.get("players", []))
    n_frames = max(counts) + 1 if counts else 0
    ge = sum(1 for i in range(n_frames) if counts.get(i, 0) >= need)
    pct = 100 * ge / max(n_frames, 1)
    dist = pd.Series([counts.get(i, 0) for i in range(n_frames)]).value_counts().sort_index()
    print(f"players: >={need} on court in {pct:.1f}% of frames "
          f"target 80 {'PASS' if pct >= 80 else 'FAIL'}")
    print("count distribution:", dict(dist))
except FileNotFoundError:
    print("players: no scene.json yet")

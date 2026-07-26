"""Accuracy vs targets, measured on the FINAL scene.json (what the viewer shows).
Usage: metrics.py OUTDIR N_PLAYERS_EXPECTED"""
import json
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "out"
need = int(sys.argv[2]) if len(sys.argv) > 2 else 12

scene = json.load(open(f"{out}/scene.json"))
frames = scene["frames"]
ts = np.array([f["t"] for f in frames])
ball_ts = np.array([f["t"] for f in frames if "ball" in f])

# in-play spans from scene ball data: gaps > 2s split rallies
spans, s = [], ball_ts[0]
for a, b in zip(ball_ts, ball_ts[1:]):
    if b - a > 2.0:
        spans.append((s, a))
        s = b
spans.append((s, ball_ts[-1]))
dt = np.median(np.diff(ts[:200]))
in_play = sum(b - a for a, b in spans) + dt * len(spans)
covered = len(ball_ts) * dt
print(f"ball: {100*covered/in_play:.1f}% of in-play time in scene "
      f"({len(spans)} rallies, {in_play:.0f}s in play)  "
      f"target 90 {'PASS' if covered/in_play >= 0.9 else 'FAIL'}")

# players: only frames that carry player data (subsampled pipelines skip frames)
counts = [len(f["players"]) for f in frames if f.get("players")]
ge = sum(1 for c in counts if c >= need)
pct = 100 * ge / max(len(counts), 1)
dist = {}
for c in counts:
    dist[c] = dist.get(c, 0) + 1
print(f"players: >={need} in {pct:.1f}% of player-frames  "
      f"target 80 {'PASS' if pct >= 80 else 'FAIL'}")
print("count distribution:", dict(sorted(dist.items())))

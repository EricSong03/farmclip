"""Cut test sub-clips + t-shifted scene JSON subsets from a processed out dir.
Usage: cut_samples.py OUTDIR NAME START_S END_S (times relative to the clip)"""
import json
import sys
from pathlib import Path

from farmclip.video import clip_video

out_dir, name, start, end = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
dst = Path("examples/samples") / name
dst.mkdir(parents=True, exist_ok=True)
clip_video(f"{out_dir}/clip.mp4", dst / "clip.mp4", start, end)
scene = json.loads(Path(f"{out_dir}/scene.json").read_text())
frames = [dict(f, t=round(f["t"] - start, 4))
          for f in scene["frames"] if start <= f["t"] <= end]
scene["frames"] = frames
(dst / "scene.json").write_text(json.dumps(scene))
n_ball = sum(1 for f in frames if "ball" in f)
print(f"{dst}: {len(frames)} frames, {n_ball} ball, "
      f"{sum(1 for f in frames if f.get('players'))} players")

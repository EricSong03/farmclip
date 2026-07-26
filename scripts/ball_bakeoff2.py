"""Bake off VballNet variants on a 2-min slice; report in-play coverage.
Also tests 30fps-subsampled input for the 60fps clip (models expect ~30fps motion).
"""
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd

VEND = Path("vendor/fast-vb-tracking")
MODELS = [
    "VballNetV1_seq9_grayscale_330_h288_w512.onnx",   # current
    "VballNetV1b_seq9_grayscale_best.onnx",
    "VballNetV1c_seq9_grayscale_best.onnx",
    "VballNetGridV1b_seq15_grayscale_20260427_194144.onnx",
]

# 2-min test slice at native 60fps + a 30fps-subsampled variant
slice_path = Path("out/tune/slice.mp4")
if not slice_path.exists():
    slice_path.parent.mkdir(parents=True, exist_ok=True)
    from farmclip.video import clip_video
    clip_video("out/mikasa/clip.mp4", slice_path, 0, 120)
    # 30fps variant: every 2nd frame
    cap = cv2.VideoCapture(str(slice_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter("out/tune/slice30.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 2 == 0:
            vw.write(f)
        i += 1
    cap.release(); vw.release()

def coverage(csv_path, fps):
    df = pd.read_csv(csv_path)
    vis = df[df.Visibility > 0].Frame.to_numpy()
    if not len(vis):
        return 0.0, 0
    spans, s = [], vis[0]
    for a, b in zip(vis, vis[1:]):
        if b - a > 2 * fps:
            spans.append((s, a)); s = b
    spans.append((s, vis[-1]))
    in_play = sum(b - a + 1 for a, b in spans)
    return 100 * len(vis) / max(in_play, 1), len(df)

for clip, fps, tag in (("out/tune/slice.mp4", 60, "60fps"), ("out/tune/slice30.mp4", 30, "30fps")):
    for m in MODELS:
        out_dir = Path(f"out/tune/{tag}_{m[:12]}")
        csvs = list(out_dir.glob("*/ball.csv"))
        if not csvs:
            subprocess.run(
                [sys.executable, "src/inference_onnx_seq_gray_v2.py",
                 "--video_path", str(Path(clip).resolve()),
                 "--model_path", str((VEND / "models" / m).resolve()),
                 "--output_dir", str(out_dir.resolve()), "--only_csv"],
                cwd=VEND, check=True, capture_output=True)
            csvs = list(out_dir.glob("*/ball.csv"))
        cov, n = coverage(csvs[0], fps)
        print(f"{tag} {m[:28]:30s} in-play {cov:.1f}%  ({n} frames)")

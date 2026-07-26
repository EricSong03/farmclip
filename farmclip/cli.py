"""farmclip CLI: full pipeline — clip -> calibrate -> ball -> players -> scene.json.

Usage: farmclip VIDEO --start 60 --end 720 --out out/mikasa
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import video
from .scene import write_scene

VBALLNET = Path(__file__).parent.parent / "vendor" / "fast-vb-tracking"
# dual-model union: Grid catches what V1c misses (4px median agreement when
# both fire; union measured +10pts in-play coverage over either alone)
VBALLNET_MODELS = [
    VBALLNET / "models" / "VballNetGridV1b_seq15_grayscale_20260427_194144.onnx",
    VBALLNET / "models" / "VballNetV1c_seq9_grayscale_best.onnx",
]


def calibrate(clip: Path, out: Path):
    """Auto-calibration: scan frames until a hypothesis passes, polish, save overlay."""
    from .lines import detect_segments
    from .hypothesis import search
    from .refine_lsq import polish
    from .calibrate import draw_overlay
    from .court import HL, HW, ATTACK, NET_H
    from .hypothesis import FAR_LINES

    info_v = video.video_info(clip)
    w, h = info_v["width"], info_v["height"]
    step = max(1, int(info_v["frames"] // 40))
    best = None
    for i, t, frame in video.read_frames(str(clip), step=step):
        segs = detect_segments(frame)

        def slope(s):
            return abs(s[3] - s[1]) / max(abs(s[2] - s[0]), 1)

        low = [s for s in segs if slope(s) < 0.15 and (s[1] + s[3]) / 2 > h * 0.6]
        band = [s for s in segs if slope(s) < 0.1 and h * 0.25 < (s[1] + s[3]) / 2 < h * 0.45
                and abs(s[2] - s[0]) > w * 0.45]
        if not band:
            continue
        # the net has TWO long parallel bands; net top = the one with a partner
        # 20-80px BELOW it (else the longest wins the tie)
        def ymid(s):
            return (s[1] + s[3]) / 2

        paired = [s for s in band
                  if any(20 < ymid(o) - ymid(s) < 80 for o in band if o is not s)]
        pool = paired or band
        net_band = min(pool, key=ymid) if paired else max(band, key=lambda s: abs(s[2] - s[0]))
        calib = None
        if low:
            attack_near = max(low, key=lambda s: abs(s[2] - s[0]))
            calib, info = search(segs, attack_near, net_band, w, h)
            if calib is not None:
                floor_pairs = [
                    (attack_near, ((ATTACK, 0, -HW), (ATTACK, 0, +HW))),
                    (info["segL"], ((-HL, 0, +HW), (HL, 0, +HW))),
                    (info["segR"], ((-HL, 0, -HW), (HL, 0, -HW))),
                    (info["segF"], FAR_LINES[info["far"]]),
                ]
                calib = polish(calib, floor_pairs, band_seg=net_band)
        if calib is None:  # head-on fallback: net plane + posts anchor
            from .hypothesis import search_headon
            calib, info = search_headon(segs, net_band, w, h)
        if calib is None:
            continue
        score = (info["n_inliers"], -calib["err"])
        if best is None or score > best[0]:
            best = (score, calib, i, frame)
            print(f"  calib candidate @frame {i}: err {calib['err']:.1f}, "
                  f"inliers {info['n_inliers']}")
    if best is None:
        sys.exit("calibration failed: no frame produced a valid hypothesis")
    _, calib, ref_frame, frame = best
    calib["ref_frame"] = ref_frame
    (out / "calib.json").write_text(json.dumps(calib, indent=1))
    dbg = out / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / "calib_overlay.jpg"), draw_overlay(frame, calib))
    print(f"calibrated on frame {ref_frame} (err {calib['err']:.1f}px) "
          f"-> {dbg / 'calib_overlay.jpg'}")
    return calib


def run_ball(clip: Path, out: Path, calib: dict, fps: float):
    """Dual VballNet union 2D -> ballistic 3D. Returns {frame: [x,y,z]}."""
    import pandas as pd
    from .ball3d import lift, fill_gaps

    dfs = []
    for i, model in enumerate(VBALLNET_MODELS):
        mdir = out / f"ball{i}"
        csvs = list(mdir.glob("*/ball.csv"))
        if not csvs:
            subprocess.run(
                [sys.executable, "src/inference_onnx_seq_gray_v2.py",
                 "--video_path", str(clip.resolve()),
                 "--model_path", str(model.resolve()),
                 "--output_dir", str(mdir.resolve()), "--only_csv"],
                cwd=VBALLNET, check=True)
            csvs = list(mdir.glob("*/ball.csv"))
        dfs.append(pd.read_csv(csvs[0]))
    # union: primary model's position wins; secondary fills its misses
    df = dfs[0]
    for other in dfs[1:]:
        n = min(len(df), len(other))
        take = (df.Visibility[:n] == 0) & (other.Visibility[:n] > 0)
        df.loc[take[take].index, ["X", "Y", "Visibility"]] = \
            other.loc[take[take].index, ["X", "Y", "Visibility"]].values
    from .ball3d import reject_outliers
    track = df[["Frame", "X", "Y", "Visibility"]].to_numpy(float)
    track = fill_gaps(reject_outliers(track), max_gap=max(3, int(fps * 0.1)))
    ball3d, n_seg, n_ok = lift(track, calib, fps)
    print(f"ball: {int((df.Visibility > 0).mean() * 100)}% detected, "
          f"{n_ok}/{n_seg} segments fitted, {len(ball3d)} 3D frames")
    return ball3d


def run_players(clip: Path, calib: dict):
    """YOLO persons -> tracks -> ground positions. Returns {frame: [players]}."""
    from ultralytics import YOLO
    from .players import Tracker, to_scene_players

    model = YOLO("yolo11n.pt")
    tracker = Tracker()
    cap = cv2.VideoCapture(str(clip))
    per_frame = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, classes=[0], conf=0.25, imgsz=960, verbose=False)[0]
        tracked = tracker.update([b.xyxy[0].tolist() for b in r.boxes])
        players = to_scene_players(tracked, calib)
        if players:
            per_frame[i] = players
        i += 1
    cap.release()
    counts = [len(v) for v in per_frame.values()]
    print(f"players: {len(per_frame)}/{i} frames, "
          f"avg on-court {sum(counts) / max(len(counts), 1):.1f}")
    return per_frame


def main(argv=None):
    ap = argparse.ArgumentParser(prog="farmclip")
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=60.0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clip = out / "clip.mp4"
    if not clip.exists():
        print(f"clipping {args.start}-{args.end}s -> {clip}")
        video.clip_video(args.video, clip, args.start, args.end)
    info = video.video_info(clip)
    fps = info["fps"]
    print(f"clip: {info['width']}x{info['height']} @ {fps:.2f}fps, {info['frames']} frames")

    calib_path = out / "calib.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else calibrate(clip, out)

    ball3d = run_ball(clip, out, calib, fps)
    players = run_players(clip, calib)

    frames = []
    for i in sorted(set(ball3d) | set(players)):
        frames.append({"t": i / fps, "ball": ball3d.get(i), "players": players.get(i, [])})
    path = write_scene(out / "scene.json", frames,
                       net_height=round(calib.get("net_h", 2.43), 2))
    print(f"scene: {path} ({len(frames)} frames, "
          f"{sum(1 for f in frames if f.get('ball'))} ball, "
          f"{sum(1 for f in frames if f.get('players'))} players)")


if __name__ == "__main__":
    main()

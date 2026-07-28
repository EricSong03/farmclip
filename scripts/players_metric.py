"""Pipeline-level acceptance for the player student: the metric the user cares
about, not raw box counts.

Runs the REAL players stage (YOLO -> IoU tracker -> floor-ray -> roster cap) on a
clip with stock yolo11s and with the fine-tuned student, and prints the same
numbers scripts/metrics.py reports: % of player-frames carrying >= N on-court
players, plus the count distribution. Crowd/bleacher boxes are filtered by the
court-margin + per-side cap, so this measures on-court recall, not detections.

Usage: players_metric.py VIDEO N_EXPECTED [--per-side 6] [--step 3]
"""
import argparse
import json
from pathlib import Path

from farmclip.cli import calibrate, run_players
from farmclip import video

STOCK = "yolo11s.pt"
FT = "finetune_out/yolo11s-vb.pt"


def dist(per_frame, need):
    counts = [len(v) for v in per_frame.values() if v]
    d = {}
    for c in counts:
        d[c] = d.get(c, 0) + 1
    ge = sum(1 for c in counts if c >= need)
    pct = 100 * ge / max(len(counts), 1)
    avg = sum(counts) / max(len(counts), 1)
    return pct, avg, dict(sorted(d.items())), len(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("need", type=int)
    ap.add_argument("--per-side", type=int, default=6)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--out", default="out/players_metric")
    args = ap.parse_args()

    out = Path(args.out) / Path(args.video).stem
    out.mkdir(parents=True, exist_ok=True)
    clip = Path(args.video)
    info = video.video_info(clip)
    fps = info["fps"]
    print(f"{clip}: {info['width']}x{info['height']} @ {fps:.2f}fps, {info['frames']} frames")

    calib_path = out / "calib.json"
    if calib_path.exists():
        calib = json.loads(calib_path.read_text())
    else:
        calib = calibrate(clip, out)
        calib_path.write_text(json.dumps(calib))

    for tag, model in (("stock", STOCK), ("finetuned", FT)):
        per_frame = run_players(clip, calib, model_name=model, per_side=args.per_side,
                                step=args.step, fps=fps)
        pct, avg, d, n = dist(per_frame, args.need)
        print(f"[{tag}] >={args.need} in {pct:.1f}% of {n} player-frames  "
              f"(avg {avg:.1f})  target 80")
        print(f"[{tag}] distribution: {d}")


if __name__ == "__main__":
    main()

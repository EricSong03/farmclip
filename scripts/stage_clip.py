"""Stage a video for manual keypoint labeling.

Extracts a ref frame, registers the run in data/runs.json (which calib.html
and build_kp_dataset.py read). Idempotent per run name.
Usage: python -m uv run python scripts/stage_clip.py <video> [run_name] [t_seconds]
"""
import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.video import video_info

ROOT = Path(__file__).parent.parent
RUNS_JSON = ROOT / "data/runs.json"
# ponytail: pre-runs.json runs, seeded once on first stage
SEED = [
    {"name": "menlo", "dir": "out", "ref": "debug/ref_frame.jpg", "video": "videos/clip.mp4"},
    {"name": "mikasa", "dir": "out/mikasa", "ref": "debug/ref_frame.jpg", "video": "videos/mikasa.mp4"},
]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    video = Path(sys.argv[1])
    if not video.exists():
        sys.exit(f"no such video: {video}")
    name = sys.argv[2] if len(sys.argv) > 2 else video.stem
    name = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    info = video_info(video)
    t = float(sys.argv[3]) if len(sys.argv) > 3 else info["frames"] / info["fps"] / 2

    run_dir = ROOT / "out" / name
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"could not read frame at t={t}s")
    ref_path = run_dir / "debug" / "ref_frame.jpg"
    cv2.imwrite(str(ref_path), frame)

    runs = json.loads(RUNS_JSON.read_text()) if RUNS_JSON.exists() else list(SEED)
    entry = {"name": name, "dir": f"out/{name}", "ref": "debug/ref_frame.jpg",
             "video": video.resolve().relative_to(ROOT).as_posix() if video.resolve().is_relative_to(ROOT) else str(video)}
    runs = [r for r in runs if r["name"] != name] + [entry]
    RUNS_JSON.write_text(json.dumps(runs, indent=1))

    print(f"staged run: {name}")
    print(f"ref frame:  {ref_path} (t={t:.1f}s)")
    print("open calib.html to label")


if __name__ == "__main__":
    main()

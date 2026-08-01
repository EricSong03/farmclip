"""Dump evenly-spaced frames from a video into data/pool for calib.html labeling.

Same bucket as the wikimedia stills: unlabeled web_NNN.jpg + a sources.txt line.
Label them in calib.html image-batch mode; build_kp_dataset.py picks them up.
Usage: python -m uv run python scripts/web_frames.py <video> [n=8]
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
WEB = ROOT / "data/pool"


def next_index():
    """Next free web_NNN slot. ponytail: read from disk, not a counter file —
    sources.txt keeps deleted indices reserved so they are never reused."""
    src = WEB / "sources.txt"
    used = {int(p.stem[4:]) for p in WEB.glob("web_*.jpg")}
    if src.exists():
        used |= {int(ln.split("\t")[0][4:-4]) for ln in
                 src.read_text(errors="replace").splitlines() if ln.startswith("web_")}
    return max(used) + 1 if used else 0


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    video = Path(sys.argv[1])
    if not video.exists():
        sys.exit(f"no such video: {video}")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    WEB.mkdir(parents=True, exist_ok=True)
    nxt = next_index()

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        sys.exit(f"could not read frame count: {video}")
    # skip first/last 5% - intros, outros, credits
    lo, hi = int(total * 0.05), int(total * 0.95)
    lines = []
    for i in range(n):
        f = lo + (hi - lo) * i // max(n - 1, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            print(f"  skip frame {f} (unreadable)")
            continue
        name = f"web_{nxt:03d}.jpg"
        cv2.imwrite(str(WEB / name), frame)
        lines.append(f"{name}\t{video.name}@frame{f}")
        nxt += 1
    cap.release()

    with src.open("a", encoding="utf-8") as fh:
        fh.write("".join(ln + "\n" for ln in lines))
    print(f"{video.name}: wrote {len(lines)} -> {lines[0].split(chr(9))[0]}..{lines[-1].split(chr(9))[0]}")


if __name__ == "__main__":
    main()

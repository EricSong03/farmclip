"""Register new labelling runs: one click-set covers a whole venue.

A run is the cheap way to label. The camera is fixed, so one annotated
reference frame labels every frame sampled from that venue -- which is why
dome1 contributes 41 training frames for a single click-set while a web still
contributes one. The venue sweep measured that density, not venue count, is
what moves held-out error, so this is the highest-yield labelling available.

Two kinds of run:

  video  -- a local file in videos/. Frames are sampled from it at build time.
  frames -- a directory of stills already pulled from a stream. Used for
            broadcast VODs, where downloading 90 minutes of video to keep 40
            frames is not worth the disk.

Both need exactly one thing from a human: annotations.json, clicked in
calib.html against the run's ref_frame.jpg.

Broadcast runs carry a risk the video runs do not: a league feed cuts to
replays, closeups and other cameras, and one annotation cannot be correct for
those frames. build_kp_dataset's drift gate is what removes them -- it drops
any frame whose court does not match the reference pose, which is the same
mechanism that already drops 35 of 43 frames on dome3. Expect a lower yield
here, and check it after labelling rather than assuming 40 frames per venue.

Usage: python -m uv run python scripts/add_runs.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

STAGE = ROOT / "out/ytstage"
RUNS_JSON = ROOT / "data/runs.json"

# name, source video (local), ref frame staged under out/ytstage/<stem>/
VIDEO_RUNS = [
    ("tallones-evl", "videos/tallones-evl.mp4", "f9749"),
    ("tallones-oos", "videos/tallones-oos-dallas.mp4", "f20708"),
    ("tallones-vlacup", "videos/tallones-vlacup.mp4", "f13330"),
    ("tallones-vlawest", "videos/tallones-vlawest.mp4", "f19805"),
]

# name, staged dir (= youtube id), ref frame. All are elevated views, the
# emptiest cell in the pool, and none appear in data/test/sources.txt.
FRAME_RUNS = [
    ("bc-lnarena", "0pXfDmucbhk", "f17281"),
    ("bc-perugia", "Z9eEWmLb0QQ", "f13889"),
    ("bc-wislutheran", "ifHkpb6XGg4", "f25272"),
    ("bc-pineview", "sOQzuu4Jenw", "f34956"),
    ("bc-nebraska", "jrP39pe-qIY", "f61811"),
    ("bc-ncaaqf", "1JMc6mQjknI", "f60358"),
]


def test_vods():
    """Video ids backing the held-out test set. Training on one voids it."""
    import re
    out = set()
    p = ROOT / "data/test/sources.txt"
    for ln in p.read_text(errors="replace").splitlines():
        if "\t" not in ln:
            continue
        url = ln.split("\t")[1]
        m = re.search(r"v=([\w-]{11})", url)
        out.add(m.group(1) if m else Path(url.split("&t=")[0]).stem)
    return out


def main():
    runs = json.loads(RUNS_JSON.read_text())
    known = {r["name"] for r in runs}
    banned = test_vods()
    added = []

    for name, video, ref in VIDEO_RUNS:
        if name in known:
            print(f"{name}: already registered")
            continue
        if not (ROOT / video).exists():
            print(f"{name}: SKIP, {video} missing")
            continue
        d = ROOT / "data/runs" / name
        d.mkdir(parents=True, exist_ok=True)
        src = STAGE / Path(video).stem / f"{ref}.jpg"
        if not src.exists():
            print(f"{name}: SKIP, ref frame {src} not staged")
            continue
        shutil.copyfile(src, d / "ref_frame.jpg")
        runs.append({"name": name, "dir": f"out/{name}", "ref": "ref_frame.jpg",
                     "video": video, "labels": f"data/runs/{name}"})
        added.append((name, "video", 1))

    for name, vid, ref in FRAME_RUNS:
        if name in known:
            print(f"{name}: already registered")
            continue
        if vid in banned:
            print(f"{name}: REFUSED, {vid} backs the held-out test set")
            continue
        sd = STAGE / vid
        frames = sorted(sd.glob("f*.jpg"))
        if not frames or not (sd / f"{ref}.jpg").exists():
            print(f"{name}: SKIP, nothing staged in {sd}")
            continue
        d = ROOT / "data/runs" / name
        (d / "frames").mkdir(parents=True, exist_ok=True)
        for f in frames:
            shutil.copyfile(f, d / "frames" / f.name)
        shutil.copyfile(sd / f"{ref}.jpg", d / "ref_frame.jpg")
        runs.append({"name": name, "dir": f"out/{name}", "ref": "ref_frame.jpg",
                     "frames": f"data/runs/{name}/frames", "source": vid,
                     "labels": f"data/runs/{name}"})
        added.append((name, "frames", len(frames)))

    RUNS_JSON.write_text(json.dumps(runs, indent=1) + "\n")
    print()
    for name, kind, n in added:
        print(f"  + {name:20} {kind:7} {n} frame(s) available")
    todo = [r["name"] for r in runs
            if not (ROOT / r.get("labels", "") / "annotations.json").exists()]
    print(f"\n{len(added)} run(s) added. Awaiting a click-set: {', '.join(todo)}")
    print("Label each in calib.html — pick the run from the dropdown, click the "
          "points on its ref frame, then rebuild the dataset.")


if __name__ == "__main__":
    main()

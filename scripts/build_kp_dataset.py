"""Build a YOLO pose dataset from manual court annotations (Phase 1, auto-calib-ai).

Stationary camera => one annotated ref frame labels every frame of its clip.
Sample ~40 frames per run, drop drifted ones (calib_eval.score_frame vs ref),
write ultralytics pose images/labels + dataset.yaml.
Usage: python -m uv run python scripts/build_kp_dataset.py
       ... --venues 20 --seed 1 --out court_v20   (venue-count ablation)

--venues builds from a random subset of N SOURCE VENUES rather than N images.
Venues, not images, because frames from one stationary camera are near
duplicates: 40 frames of dome1 are one viewpoint forty times, so an
image-count curve would measure the wrong axis. Train a model per N, score
each with scripts/benchmark.py, and the curve says whether more venues still
buys accuracy or whether it has flattened -- which is the question "do we need
more data" actually reduces to.
"""
import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.calibrate import solve_labeled
from farmclip.court import KEYPOINTS
from farmclip.video import video_info, read_frames
from calib_eval import score_frame

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--venues", type=int, help="use only N source venues")
_ap.add_argument("--seed", type=int, default=0, help="which N venues")
_ap.add_argument("--out", default="court", help="subdir under data/dataset/")
_ap.add_argument("--path", help="value for dataset.yaml's `path:` (default: this "
                                "machine's absolute path). Set it to the GPU "
                                "box's mount so the yaml needs no hand-editing.")
ARGS = _ap.parse_args()

ROOT = Path(__file__).parent.parent
if (ROOT / "data/runs.json").exists():
    # labels/ not dir/: annotations, calib and the ref frame are tracked data
    # and live under data/runs/<name>/. dir/ is the pipeline's working output
    # directory, which is gitignored and may not exist on a fresh clone.
    RUNS = [(r["name"], ROOT / r.get("labels", r["dir"]), ROOT / r["video"])
            for r in json.loads((ROOT / "data/runs.json").read_text())]
else:  # fallback: pre-runs.json layout
    RUNS = [("menlo", ROOT / "out", ROOT / "videos/clip.mp4"),
            ("mikasa", ROOT / "out/mikasa", ROOT / "videos/mikasa.mp4")]
OUT = ROOT / "data/dataset" / ARGS.out
NAMES = list(KEYPOINTS)  # canonical 18-slot order


def drop_posts(a):
    """ponytail: post geometry varies per venue — excluded from training/solve."""
    return {k: v for k, v in a.items() if not k.startswith("post_")}


def court_median(img, calib):
    """(median court-line err, n court matches); net excluded."""
    errs = [v for k, v in score_frame(img, calib).items() if k != "net"]
    return (float(np.median(errs)) if errs else None, len(errs))


def label_line(ann, w, h):
    """YOLO pose row: class cx cy bw bh + 18x (x y v), normalized."""
    kps, vis_uv = [], []
    for name in NAMES:
        uv = ann.get(name)
        if uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h:
            kps += [uv[0] / w, uv[1] / h, 2]
            vis_uv.append(uv)
        else:
            kps += [0, 0, 0]
    us, vs = [p[0] for p in vis_uv], [p[1] for p in vis_uv]
    px, py = (max(us) - min(us)) * 0.05, (max(vs) - min(vs)) * 0.05
    x0, x1 = max(0, min(us) - px), min(w - 1, max(us) + px)
    y0, y1 = max(0, min(vs) - py), min(h - 1, max(vs) + py)
    box = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    return "0 " + " ".join(f"{v:.6f}" for v in box + kps)


# Wipe before writing. This script only ever ADDED files, so anything a later
# run excluded stayed on disk -- and dataset.yaml points at the DIRECTORY, so
# training consumed it regardless of what this script reported. Measured at the
# time this was found: the builder reported 125 web images and the directory
# held 160, the extra 35 being frames deleted during labelling, frames rejected
# by the quality gates, and the 12 leaked test-venue frames. Every model
# trained here had been fed images its own gates had rejected.
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    p = OUT / sub
    if p.exists():
        for f in p.iterdir():
            if f.is_file():
                f.unlink()
    p.mkdir(parents=True, exist_ok=True)

# web-scraped stills labeled in calib.html image-batch mode
webdir = ROOT / "data/pool"


def _source_map(pool):
    """image stem -> source video id, from a pool's sources.txt."""
    out = {}
    sp = pool / "sources.txt"
    if not sp.exists():
        return out
    for ln in sp.read_text(errors="replace").splitlines():
        if "\t" not in ln:
            continue
        name, url = ln.split("\t")[:2]
        m = re.search(r"v=([\w-]{11})|youtu\.be/([\w-]{11})", url)
        vid = (m.group(1) or m.group(2)) if m else Path(url.split("&t=")[0].split("@")[0]).stem
        out[Path(name).stem] = vid
    return out


# A held-out set is only held out if none of its VIDEOS are trained on. Frames
# from a stationary camera are near-duplicates of each other, so one training
# frame from a test venue leaks that venue's exact viewpoint. This was not
# hypothetical: tallones-usav and tallones-vladimes were reserved as test
# venues while 12 of their frames sat in the training pool from an earlier
# session, which silently turned the whole target-PoV benchmark group into a
# seen-venue score.
TEST_SOURCES = set(_source_map(ROOT / "data/test").values())
WEB_SOURCES = _source_map(webdir)


# ---- venue selection -------------------------------------------------------
# Only venues that actually contribute a LABELLED image count. sources.txt
# also lists frames staged but not yet clicked, and sampling those would
# silently shrink the split instead of holding venue count constant.
_labelled = {q.stem for q in (webdir / "labels").glob("*.json")}
ALL_VENUES = sorted({v for k, v in WEB_SOURCES.items()
                     if k in _labelled and v not in TEST_SOURCES}
                    | {r[0] for r in RUNS})
if ARGS.venues:
    _rng = random.Random(ARGS.seed)
    VENUES = set(_rng.sample(ALL_VENUES, min(ARGS.venues, len(ALL_VENUES))))
    print(f"[ablation] {len(VENUES)} of {len(ALL_VENUES)} venues, seed {ARGS.seed}")
else:
    VENUES = set(ALL_VENUES)

kept_all, n_kept = [], 0
for run, outdir, video in RUNS:
    if run not in VENUES:
        continue
    if not (outdir / "annotations.json").exists():
        print(f"[{run}] no annotations.json — skipping (label it in calib.html)")
        continue
    ann = drop_posts(json.loads((outdir / "annotations.json").read_text()))
    # Quality gate on the run's own labels. Clicks are trusted now (no mirror
    # search to rescue a bad set), so a run whose annotations do not solve to a
    # physical camera would train the model on nonsense. menlo fails this at
    # 741px as-clicked and 315px mirrored -- broken under either convention --
    # and its 30px calib is what poisoned the lineseg masks too.
    try:
        _ref = cv2.imread(str(outdir / "ref_frame.jpg"))
        _c = solve_labeled(ann, _ref.shape[1], _ref.shape[0])
        if _c["cam_below_floor"] or _c["err"] > 25:
            print(f"[{run}] SKIPPED - labels solve to {_c['err']:.0f}px"
                  f"{' with camera under the floor' if _c['cam_below_floor'] else ''}"
                  f" - relabel in calib.html")
            continue
        print(f"[{run}] labels ok: {_c['err']:.1f}px, cam {_c['cam_height']}m up")
    except Exception as _e:
        print(f"[{run}] SKIPPED - label solve failed ({_e})")
        continue
    calib = None
    if (outdir / "calib.json").exists():
        calib = json.loads((outdir / "calib.json").read_text())
        # Clicks are taken as given. consistent_names/canonical_lr used to
        # live here to undo court.py's inverted L/R convention; that is
        # fixed at the source now, so renaming would only reintroduce the
        # ambiguity it was compensating for.
        ref = cv2.imread(str(outdir / "ref_frame.jpg"))
        ref_med, _ = court_median(ref, calib)
        print(f"[{run}] ref median {ref_med}")
    else:
        print(f"[{run}] no calib.json — drift gate off, keeping all sampled frames")
    if not video.exists():
        # videos/ is untracked and mostly absent on other machines, so a clone
        # silently builds a web-only dataset that is NOT comparable to one built
        # here. Say so loudly rather than producing a quietly different split.
        print(f"[{run}] VIDEO MISSING ({video.name}) - this run contributes "
              f"NOTHING. A dataset built without it is not comparable to one "
              f"built where the videos exist.")
        continue
    step = max(1, video_info(video)["frames"] // 40)
    kept = dropped = 0
    for i, t, frame in read_frames(video, step=step):
        if calib is None:
            ok = True
        else:
            med, n = court_median(frame, calib)
            if ref_med:  # ponytail: 2x-ref drift gate; tune if a clip pans
                ok = med is not None and med <= 2 * ref_med
            else:
                ok = n >= 4
        if not ok:
            dropped += 1
            continue
        split = "val" if n_kept % 5 == 4 else "train"
        n_kept += 1
        name = f"{run}_{i:05d}"
        h, w = frame.shape[:2]
        cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), frame)
        (OUT / "labels" / split / f"{name}.txt").write_text(label_line(ann, w, h) + "\n")
        kept_all.append((split, name, ann))
        kept += 1
    print(f"[{run}] kept {kept} dropped {dropped}")

web_kept, web_leaked = 0, []
for lp in sorted((webdir / "labels").glob("*.json")) if (webdir / "labels").exists() else []:
    if WEB_SOURCES.get(lp.stem) in TEST_SOURCES:
        web_leaked.append(f"{lp.stem} ({WEB_SOURCES[lp.stem]})")
        continue
    if WEB_SOURCES.get(lp.stem) not in VENUES:
        continue
    imgp = next((p for e in (".jpg", ".jpeg", ".png") if (p := webdir / (lp.stem + e)).exists()), None)
    if imgp is None:
        print(f"[web] {lp.stem}: no image — skipping")
        continue
    ann = drop_posts(json.loads(lp.read_text()))
    if len(ann) < 5:
        print(f"[web] {lp.stem}: only {len(ann)} points — skipping")
        continue
    frame = cv2.imread(str(imgp))
    h, w = frame.shape[:2]
    try:
        calib = solve_labeled(ann, w, h)
        if calib["err"] > 30:  # floor-only err: tighter gate than the old joint solve
            print(f"[web] {lp.stem}: floor err {calib['err']:.0f}px — rejecting (bad labels?)")
            continue
        pass  # clicks taken as given (see note above)
    except Exception as e:
        print(f"[web] {lp.stem}: solve failed ({e}) — keeping points as clicked")
    split = "val" if n_kept % 5 == 4 else "train"
    n_kept += 1
    name = f"web_{lp.stem}"
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), frame)
    (OUT / "labels" / split / f"{name}.txt").write_text(label_line(ann, w, h) + "\n")
    kept_all.append((split, name, ann))
    web_kept += 1
if web_leaked:
    print(f"[web] EXCLUDED {len(web_leaked)} frame(s) whose video is in the held-out "
          f"test set: {', '.join(web_leaked)}")
if web_kept:
    print(f"[web] kept {web_kept}")

train = sum(1 for s, *_ in kept_all if s == "train")
print(f"total: {train} train / {len(kept_all) - train} val ({web_kept} web)")

# flip_idx: horizontal flip swaps _left <-> _right
flip_idx = [NAMES.index(n.replace("_left", "_R").replace("_right", "_left").replace("_R", "_right"))
            for n in NAMES]
print("flip_idx:", {NAMES[i]: NAMES[j] for i, j in enumerate(flip_idx)})

_yaml_path = ARGS.path or OUT.resolve().as_posix()
(OUT / "dataset.yaml").write_text(
    f"path: {_yaml_path}\ntrain: images/train\nval: images/val\n"
    f"kpt_shape: [18, 3]\nflip_idx: {flip_idx}\nnames:\n  0: court\n")
(OUT / "kpt_names.json").write_text(json.dumps(NAMES, indent=1))

# visual spot-check
random.seed(0)
for split, name, ann in random.sample(kept_all, min(3, len(kept_all))):
    img = cv2.imread(str(OUT / "images" / split / f"{name}.jpg"))
    vis = [ann[n] for n in NAMES if n in ann]
    us, vs = [p[0] for p in vis], [p[1] for p in vis]
    cv2.rectangle(img, (int(min(us)), int(min(vs))), (int(max(us)), int(max(vs))), (0, 255, 255), 1)
    for n in NAMES:
        if n in ann:
            u, v = int(ann[n][0]), int(ann[n][1])
            cv2.circle(img, (u, v), 4, (0, 0, 255), -1)
            cv2.putText(img, n, (u + 5, v - 5), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)
    cv2.imwrite(str(OUT / f"check_{name}.jpg"), img)
    print(f"check_{name}.jpg")

"""Build a YOLO pose dataset from manual court annotations (Phase 1, auto-calib-ai).

Stationary camera => one annotated ref frame labels every frame of its clip.
Sample ~40 frames per run, drop drifted ones (calib_eval.score_frame vs ref),
write ultralytics pose images/labels + dataset.yaml.
Usage: python -m uv run python scripts/build_kp_dataset.py
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.calibrate import consistent_names, solve_web
from farmclip.court import KEYPOINTS
from farmclip.video import video_info, read_frames
from calib_eval import score_frame

ROOT = Path(__file__).parent.parent
if (ROOT / "out/runs.json").exists():
    RUNS = [(r["name"], ROOT / r["dir"], ROOT / r["video"])
            for r in json.loads((ROOT / "out/runs.json").read_text())]
else:  # fallback: pre-runs.json layout
    RUNS = [("menlo", ROOT / "out", ROOT / "videos/clip.mp4"),
            ("mikasa", ROOT / "out/mikasa", ROOT / "videos/mikasa.mp4")]
OUT = ROOT / "out/finetune/court"
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


for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

kept_all, n_kept = [], 0
for run, outdir, video in RUNS:
    if not (outdir / "annotations.json").exists():
        print(f"[{run}] no annotations.json — skipping (label it in calib.html)")
        continue
    ann = drop_posts(json.loads((outdir / "annotations.json").read_text()))
    calib = None
    if (outdir / "calib.json").exists():
        calib = json.loads((outdir / "calib.json").read_text())
        ann = consistent_names(ann, calib)
        ref = cv2.imread(str(outdir / "debug" / "ref_frame.jpg"))
        ref_med, _ = court_median(ref, calib)
        print(f"[{run}] ref median {ref_med}")
    else:
        print(f"[{run}] no calib.json — drift gate off, keeping all sampled frames")
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

# web-scraped stills labeled in calib.html image-batch mode
webdir = ROOT / "out/webimgs"
web_kept = 0
for lp in sorted((webdir / "labels").glob("*.json")) if (webdir / "labels").exists() else []:
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
        calib, _net_h = solve_web(ann, w, h)
        if calib["err"] > 30:  # floor-only err: tighter gate than the old joint solve
            print(f"[web] {lp.stem}: floor err {calib['err']:.0f}px — rejecting (bad labels?)")
            continue
        ann = consistent_names(ann, calib)
    except Exception as e:
        print(f"[web] {lp.stem}: solve failed ({e}) — keeping points as clicked")
    split = "val" if n_kept % 5 == 4 else "train"
    n_kept += 1
    name = f"web_{lp.stem}"
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), frame)
    (OUT / "labels" / split / f"{name}.txt").write_text(label_line(ann, w, h) + "\n")
    kept_all.append((split, name, ann))
    web_kept += 1
if web_kept:
    print(f"[web] kept {web_kept}")

train = sum(1 for s, *_ in kept_all if s == "train")
print(f"total: {train} train / {len(kept_all) - train} val ({web_kept} web)")

# flip_idx: horizontal flip swaps _left <-> _right
flip_idx = [NAMES.index(n.replace("_left", "_R").replace("_right", "_left").replace("_R", "_right"))
            for n in NAMES]
print("flip_idx:", {NAMES[i]: NAMES[j] for i, j in enumerate(flip_idx)})

(OUT / "dataset.yaml").write_text(
    f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
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

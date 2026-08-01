"""Build a court-line segmentation dataset (Phase 1, line-seg-calib plan).

Masks come free from solved calibs: render each court line through the calib
as a 5px polyline into a class-index PNG. Sources: video runs (calib.json x
sampled frames, same drift gate as build_kp_dataset) + web stills (solve_web).
Usage: python -m uv run python scripts/build_lineseg_dataset.py
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.calibrate import calib_matrices, project, solve_web
from farmclip.court import HL, HW, ATTACK, NET_H
from farmclip.lineseg import CLASSES, NET_EXT, class_lines
from farmclip.video import video_info, read_frames
from calib_eval import score_frame

ROOT = Path(__file__).parent.parent
RUNS = [(r["name"], ROOT / r["dir"], ROOT / r["video"])
        for r in json.loads((ROOT / "data/runs.json").read_text())]
OUT = ROOT / "data/dataset/lineseg"
def render_mask(calib, w, h, net_h):
    """uint8 (h,w) class-index map; segments behind camera / far offscreen clipped."""
    mask = np.zeros((h, w), np.uint8)
    _, rvec, tvec = calib_matrices(calib)
    R = cv2.Rodrigues(rvec)[0]
    for cid, (a, b) in class_lines(net_h).items():
        p3 = np.linspace(a, b, 200)
        z_cam = (p3 @ R.T + tvec.ravel())[:, 2]
        uv = project(calib, p3)
        ok = (z_cam > 0.1) & np.isfinite(uv).all(1) \
            & (np.abs(uv[:, 0]) < 4 * w) & (np.abs(uv[:, 1]) < 4 * h)
        for i in range(len(p3) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(mask, tuple(uv[i].astype(int)), tuple(uv[i + 1].astype(int)),
                         cid, 5)
    return mask


def court_median(img, calib):
    errs = [v for k, v in score_frame(img, calib).items() if k != "net"]
    return float(np.median(errs)) if errs else None, len(errs)


for sub in ("images/train", "images/val", "masks/train", "masks/val"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

kept_all, n_kept = [], 0


def emit(name, frame, calib, net_h):
    global n_kept
    mask = render_mask(calib, frame.shape[1], frame.shape[0], net_h)
    if (mask > 0).sum() < 500:  # calib projects the court entirely offscreen
        return False
    split = "val" if n_kept % 5 == 4 else "train"
    n_kept += 1
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), frame)
    cv2.imwrite(str(OUT / "masks" / split / f"{name}.png"), mask)
    kept_all.append((split, name))
    return True


for run, outdir, video in RUNS:
    if not ((outdir / "annotations.json").exists() and (outdir / "calib.json").exists()):
        print(f"[{run}] missing annotations/calib — skipping")
        continue
    calib = json.loads((outdir / "calib.json").read_text())
    net_h = calib.get("net_h_est") or NET_H
    ref = cv2.imread(str(outdir / "debug" / "ref_frame.jpg"))
    ref_med, _ = court_median(ref, calib) if ref is not None else (None, 0)
    print(f"[{run}] ref median {ref_med} net_h {net_h}")
    step = max(1, video_info(video)["frames"] // 30)
    kept = dropped = 0
    for i, t, frame in read_frames(video, step=step):
        if ref_med:  # ponytail: 2x-ref drift gate, same as build_kp_dataset
            med, _ = court_median(frame, calib)
            if med is None or med > 2 * ref_med:
                dropped += 1
                continue
        kept += emit(f"{run}_{i:05d}", frame, calib, net_h)
    print(f"[{run}] kept {kept} dropped {dropped}")
run_kept = n_kept

webdir = ROOT / "data/pool"
web_kept = 0
for lp in sorted((webdir / "labels").glob("*.json")) if (webdir / "labels").exists() else []:
    imgp = next((p for e in (".jpg", ".jpeg", ".png") if (p := webdir / (lp.stem + e)).exists()), None)
    if imgp is None:
        continue
    clicks = {k: v for k, v in json.loads(lp.read_text()).items()
              if not k.startswith("post_")}
    frame = cv2.imread(str(imgp))
    h, w = frame.shape[:2]
    try:
        calib, net_h = solve_web(clicks, w, h)
    except Exception as e:
        print(f"[web] {lp.stem}: solve failed ({e}) — skipping")
        continue
    if calib["err"] > 30:
        print(f"[web] {lp.stem}: floor err {calib['err']:.0f}px — skipping")
        continue
    web_kept += emit(f"web_{lp.stem}", frame, calib, net_h or NET_H)
print(f"[web] kept {web_kept}")

train = sum(1 for s, _ in kept_all if s == "train")
print(f"total: {train} train / {len(kept_all) - train} val "
      f"({run_kept} run frames, {web_kept} web)")
(OUT / "meta.json").write_text(json.dumps({"classes": CLASSES}, indent=1))

# visual spot-check: mask color-coded over image
COLORS = np.array([(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0),
                   (255, 255, 0), (255, 0, 255), (0, 128, 255), (255, 255, 255)], np.uint8)
random.seed(0)
for split, name in random.sample(kept_all, min(3, len(kept_all))):
    img = cv2.imread(str(OUT / "images" / split / f"{name}.jpg"))
    mask = cv2.imread(str(OUT / "masks" / split / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    vis = np.where((mask > 0)[..., None], COLORS[mask], img)
    cv2.imwrite(str(OUT / f"check_{name}.jpg"), vis)
    print(f"check_{name}.jpg")

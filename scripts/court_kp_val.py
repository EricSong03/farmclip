"""Per-keypoint accuracy review for the court-keypoint pose model, on the
train-pack's own val split (self-contained -- needs no clicked annotations).

Runs the model over court/images/val, matches against the YOLO-format GT
labels, and reports per-keypoint pixel error + PCK. Draws pred (green x) vs
GT (red +) overlays and a static index.html into out/debug/court_kp_val/.

Usage: python scripts/court_kp_val.py [weights.pt] [dataset_dir] [out_tag]
"""
import html
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
COURT = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "court")
OUT = ROOT / "out/debug" / (sys.argv[3] if len(sys.argv) > 3 else "court_kp_val")
NAMES = json.loads((ROOT / "court/kpt_names.json").read_text())
CONF = 0.25
# PCK thresholds as a fraction of the image diagonal.
PCK_FRACS = (0.01, 0.02, 0.05)

weights = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/pose/court/weights/best.pt"
if not weights.exists():
    sys.exit(f"weights not found: {weights}")
model = YOLO(str(weights))

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

per = {n: {"errs": [], "norm": [], "gt": 0, "missed": 0} for n in NAMES}
cards, n_nodet = [], 0
img_paths = sorted((COURT / "images/val").glob("*.jpg"))

for ip in img_paths:
    lp = COURT / "labels/val" / (ip.stem + ".txt")
    img = cv2.imread(str(ip))
    H, W = img.shape[:2]
    diag = float(np.hypot(W, H))

    # GT: class cx cy w h then 18 * (x, y, v), all normalised.
    vals = [float(v) for v in lp.read_text().split()]
    gt = np.array(vals[5:], dtype=float).reshape(18, 3)
    gt[:, 0] *= W
    gt[:, 1] *= H

    res = model.predict(str(ip), conf=CONF, verbose=False)[0]
    if res.keypoints is None or len(res.boxes) == 0:
        n_nodet += 1
        pred = None
    else:
        best = int(np.argmax(res.boxes.conf.cpu().numpy()))
        pred = res.keypoints.data.cpu().numpy()[best]  # (18, 3) -> x, y, conf

    vis = img.copy()
    rows = []
    for i, name in enumerate(NAMES):
        gx, gy, gv = gt[i]
        if gv <= 0:
            continue  # not labelled -> excluded, same as the pose loss does
        per[name]["gt"] += 1
        cv2.drawMarker(vis, (int(gx), int(gy)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        if pred is None:
            per[name]["missed"] += 1
            rows.append((name, None))
            continue
        px, py, pc = pred[i]
        err = float(np.hypot(px - gx, py - gy))
        per[name]["errs"].append(err)
        per[name]["norm"].append(err / diag)
        cv2.drawMarker(vis, (int(px), int(py)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.line(vis, (int(gx), int(gy)), (int(px), int(py)), (0, 255, 255), 1)
        rows.append((name, err))

    errs = [e for _, e in rows if e is not None]
    med = float(np.median(errs)) if errs else float("nan")
    cv2.putText(vis, f"{ip.stem}  median err {med:.1f}px", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(str(OUT / f"{ip.stem}.jpg"), vis)
    cards.append((ip.stem, med, len(errs), diag))

# ---- report ----
print(f"weights: {weights}")
print(f"val images: {len(img_paths)}   images with no detection: {n_nodet}\n")
print(f"{'keypoint':<18} {'n_gt':>5} {'miss':>5} {'median':>8} {'mean':>8} {'p90':>8}   " +
      "  ".join(f"PCK@{f:g}" for f in PCK_FRACS))
overall, overall_n = [], []
for name in NAMES:
    d = per[name]
    if d["gt"] == 0:
        print(f"{name:<18} {0:>5} {'-':>5} {'(unlabelled)':>8}")
        continue
    e = np.array(d["errs"]) if d["errs"] else np.array([np.nan])
    overall.extend(d["errs"]); overall_n.extend(d["norm"])
    nz = np.array(d["norm"]) if d["norm"] else np.array([np.nan])
    pcks = "  ".join(f"{100 * np.mean(nz <= f):>7.1f}%" for f in PCK_FRACS)
    print(f"{name:<18} {d['gt']:>5} {d['missed']:>5} {np.median(e):>8.1f} "
          f"{np.mean(e):>8.1f} {np.percentile(e, 90):>8.1f}   {pcks}")

o = np.array(overall); on = np.array(overall_n)
print(f"\nALL keypoints: n={len(o)}  median={np.median(o):.1f}px  mean={np.mean(o):.1f}px  "
      f"p90={np.percentile(o, 90):.1f}px")
for f in PCK_FRACS:
    print(f"  PCK@{f:g} of image diag: {100 * np.mean(on <= f):.1f}%")

cards.sort(key=lambda c: -(c[1] if c[1] == c[1] else 1e9))
body = "".join(
    f'<div class=c><h3>{html.escape(s)} &mdash; median {m:.1f}px ({n} kpts)</h3>'
    f'<img src="{html.escape(s)}.jpg"></div>' for s, m, n, _ in cards)
(OUT / "index.html").write_text(
    "<style>body{background:#111;color:#eee;font:14px system-ui}.c{margin:18px 0}"
    "img{max-width:100%;border:1px solid #333}</style>"
    f"<h1>court keypoint val &mdash; {html.escape(str(weights))}</h1>"
    "<p>green &times; = pred, red + = GT, worst frames first</p>" + body)
print(f"\ngallery: {OUT / 'index.html'}")

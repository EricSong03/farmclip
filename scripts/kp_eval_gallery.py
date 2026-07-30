"""Regenerable accuracy-review gallery for the court-keypoint pose model.

Samples 8 frames/clip, runs YOLO pose, draws pred (green) vs GT clicks (red +),
writes jpgs + a static index.html to out/debug/kp_eval/.
Usage: python -m uv run python scripts/kp_eval_gallery.py [model.pt]
"""
import html
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
RUNS = [("menlo", ROOT / "videos/clip.mp4", ROOT / "out/annotations.json"),
        ("mikasa", ROOT / "videos/mikasa.mp4", ROOT / "out/mikasa/annotations.json")]
OUT = ROOT / "out/debug/kp_eval"
NAMES = json.loads((ROOT / "out/finetune/court/kpt_names.json").read_text())
CONF = 0.3
N_FRAMES = 8

model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
    p for p in (ROOT / "runs/pose/finetune_out/court-pose/weights/best.pt",
                ROOT / "finetune_out/court-pose/weights/best.pt") if p.exists())
model = YOLO(str(model_path))

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

per_kpt = {n: {"errs": [], "confs": [], "det": 0, "frames": 0} for n in NAMES}
run_stats, cards = {}, []

from farmclip.calibrate import consistent_names

for run, video, ann_path in RUNS:
    ann = json.loads(ann_path.read_text())
    calib_path = ann_path.parent / "calib.json"
    if calib_path.exists():  # same L/R name correction as training labels
        ann = consistent_names(ann, json.loads(calib_path.read_text()))
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int((i + 0.5) * total / N_FRAMES) for i in range(N_FRAMES)]
    errs_run, ndet_run = [], []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        r = model.predict(frame, imgsz=1280, verbose=False)[0]
        preds = {}  # name -> (u, v, conf), best-box detection only
        if r.keypoints is not None and len(r.keypoints) > 0:
            bi = int(np.argmax(r.boxes.conf.cpu().numpy())) if len(r.boxes) else 0
            xy = r.keypoints.xy.cpu().numpy()[bi]
            kc = (r.keypoints.conf.cpu().numpy()[bi]
                  if r.keypoints.conf is not None else np.ones(len(NAMES)))
            for k, name in enumerate(NAMES):
                if kc[k] >= CONF:
                    preds[name] = (float(xy[k][0]), float(xy[k][1]), float(kc[k]))
        errs = []
        n_swapped = 0
        for name in NAMES:
            per_kpt[name]["frames"] += 1
            gt = ann.get(name)
            p = preds.get(name)
            if p:
                per_kpt[name]["det"] += 1
                per_kpt[name]["confs"].append(p[2])
            if gt:
                g = (int(gt[0]), int(gt[1]))
                cv2.drawMarker(frame, g, (0, 0, 255), cv2.MARKER_CROSS, 12, 1)
            if p and gt:
                e = float(np.hypot(p[0] - gt[0], p[1] - gt[1]))
                # side-agnostic: a name-swapped L/R pair is fixed downstream by
                # solve_auto, so score against the better of gt[name]/gt[mirror]
                mirror = name.replace("_left", "_R").replace("_right", "_left").replace("_R", "_right")
                gm = ann.get(mirror)
                if gm is not None:
                    em = float(np.hypot(p[0] - gm[0], p[1] - gm[1]))
                    if em < e:
                        e = em
                        n_swapped += 1
                errs.append(e)
                per_kpt[name]["errs"].append(e)
                cv2.line(frame, (int(p[0]), int(p[1])), (int(gt[0]), int(gt[1])),
                         (0, 255, 255), 1)
            if p:
                u, v = int(p[0]), int(p[1])
                cv2.circle(frame, (u, v), 3, (0, 255, 0), -1)
                cv2.putText(frame, f"{name} {p[2]:.2f}", (u + 4, v - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
        fname = f"{run}_{fi:05d}.jpg"
        cv2.imwrite(str(OUT / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        errs_run += errs
        ndet_run.append(len(preds))
        mean_e = f"{np.mean(errs):.1f}" if errs else "-"
        swap_note = f" | {n_swapped} L/R-swapped" if n_swapped else ""
        cards.append((fname, f"{fname} | mean err {mean_e} px | {len(preds)} kpts{swap_note}"))
    cap.release()
    run_stats[run] = {
        "mean": float(np.mean(errs_run)) if errs_run else None,
        "median": float(np.median(errs_run)) if errs_run else None,
        "matched": len(errs_run),
        "det_avg": float(np.mean(ndet_run)) if ndet_run else 0.0,
    }
    print(f"[{run}] mean {run_stats[run]['mean']:.1f} px, "
          f"median {run_stats[run]['median']:.1f} px, "
          f"{run_stats[run]['matched']} matched, "
          f"{run_stats[run]['det_avg']:.1f} kpts/frame"
          if errs_run else f"[{run}] no matched keypoints")

# unseen external images (no GT): pred overlays only, for generalization check
EXT = ROOT / "out/finetune/court/testimgs"
ext_cards = []
for img_path in sorted(EXT.glob("*.jpg")) + sorted(EXT.glob("*.png")) if EXT.exists() else []:
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue
    r = model.predict(frame, imgsz=1280, verbose=False)[0]
    confs = []
    if r.keypoints is not None and len(r.keypoints) > 0:
        bi = int(np.argmax(r.boxes.conf.cpu().numpy())) if len(r.boxes) else 0
        xy = r.keypoints.xy.cpu().numpy()[bi]
        kc = (r.keypoints.conf.cpu().numpy()[bi]
              if r.keypoints.conf is not None else np.ones(len(NAMES)))
        for k, name in enumerate(NAMES):
            if kc[k] >= CONF:
                confs.append(float(kc[k]))
                u, v = int(xy[k][0]), int(xy[k][1])
                cv2.circle(frame, (u, v), 3, (0, 255, 0), -1)
                cv2.putText(frame, f"{name} {kc[k]:.2f}", (u + 4, v - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
    fname = f"ext_{img_path.stem}.jpg"
    cv2.imwrite(str(OUT / fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    mc = f"{np.mean(confs):.2f}" if confs else "-"
    ext_cards.append((fname, f"{img_path.name} | {len(confs)} kpts >={CONF} | mean conf {mc}"))
if ext_cards:
    print(f"[ext] {len(ext_cards)} unseen images -> ext_*.jpg overlays")

# worst-3 names by median error (global; per-run split not worth it yet)
by_med = sorted(((n, float(np.median(d["errs"]))) for n, d in per_kpt.items() if d["errs"]),
                key=lambda x: -x[1])
worst3 = ", ".join(f"{n} ({e:.0f}px)" for n, e in by_med[:3])

mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(model_path.stat().st_mtime))
now = time.strftime("%Y-%m-%d %H:%M:%S")

rows = "".join(
    f"<tr><td>{r}</td><td>{s['mean']:.1f}</td><td>{s['median']:.1f}</td>"
    f"<td>{s['det_avg']:.1f}</td><td>{s['matched']}</td></tr>"
    if s["mean"] is not None else
    f"<tr><td>{r}</td><td colspan=4>no matches</td></tr>"
    for r, s in run_stats.items())

kpt_rows = ""
for n in NAMES:
    d = per_kpt[n]
    med = f"{np.median(d['errs']):.1f}" if d["errs"] else "-"
    conf = f"{np.median(d['confs']):.2f}" if d["confs"] else "-"
    rate = d["det"] / d["frames"] if d["frames"] else 0
    kpt_rows += (f"<tr><td>{n}</td><td>{med}</td>"
                 f"<td>{rate:.0%}</td><td>{conf}</td></tr>")

ext_imgs = "".join(
    f'<a href="{f}"><figure><img src="{f}" loading="lazy">'
    f"<figcaption>{html.escape(c)}</figcaption></figure></a>" for f, c in ext_cards)
ext_section = (f'<h2>Unseen courts (generalization)</h2><div class="grid">{ext_imgs}</div>'
               if ext_cards else "")

imgs = "".join(
    f'<a href="{f}"><figure><img src="{f}" loading="lazy">'
    f"<figcaption>{html.escape(c)}</figcaption></figure></a>" for f, c in cards)

(OUT / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>kp_eval gallery</title>
<style>
body{{background:#111;color:#ddd;font:14px sans-serif;margin:20px}}
table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #444;padding:4px 10px;text-align:left}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:10px}}
figure{{margin:0}} img{{width:100%;display:block}}
figcaption{{font-size:12px;color:#aaa;padding:2px 0}}
a{{color:inherit;text-decoration:none}}
</style>
<h1>Court keypoint eval</h1>
<p>generated {now} · model {model_path} · checkpoint mtime {mtime}</p>
<h2>Per-run summary</h2>
<table><tr><th>run</th><th>mean err px</th><th>median err px</th>
<th>kpts/frame &ge;{CONF}</th><th>matched</th></tr>{rows}</table>
<p>worst 3 by median error: {worst3 or "-"}</p>
<h2>Per-keypoint</h2>
<table><tr><th>name</th><th>median err px</th><th>det rate</th><th>median conf</th></tr>
{kpt_rows}</table>
<h2>Frames</h2><div class="grid">{imgs}</div>
{ext_section}
""", encoding="utf-8")

print((OUT / "index.html").resolve())

"""Render one model's court fit on TRAIN and TEST frames, side by side, as HTML.

The tables say v9 fits what it trained on and misses what it did not. This is
that claim as pictures, which is the only form of it anyone has ever been able
to check quickly.

Each tile draws the model's projected court (cyan) over the frame, with the
hand-clicked ground-truth points as green crosses. The caption gives the error
against the ground-truth POSE, not against the model's own detections -- a pose
can fit its own keypoints tightly and still be hundreds of pixels from the
court, which is the failure this whole benchmark exists to catch.

Usage: python -m uv run python scripts/v9_gallery.py [out.html] [--model X.onnx] [--n 20]
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark import compare, gt_reject  # noqa: E402
from farmclip.calibrate import draw_overlay, solve_labeled  # noqa: E402
from farmclip.court import KEYPOINTS  # noqa: E402
from farmclip.kp_detect import detect_keypoints  # noqa: E402

NAMES = list(KEYPOINTS)
TILE_W = 620


def label_points(txt_path, w, h):
    """YOLO pose row -> {keypoint name: (u, v)} in pixels, visible points only."""
    parts = Path(txt_path).read_text().split()
    kps = [float(x) for x in parts[5:]]
    out = {}
    for i, name in enumerate(NAMES):
        x, y, v = kps[3 * i], kps[3 * i + 1], kps[3 * i + 2]
        if v > 0:
            out[name] = (x * w, y * h)
    return out


def tile(img_path, clicks, model, note):
    """(html, err) for one frame, or (None, None) if it cannot be scored."""
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None, None
    h, w = frame.shape[:2]
    gt = solve_labeled(clicks, w, h)
    kp = detect_keypoints(frame, model_path=model)
    if len(kp) < 4:
        cand, err = None, None
    else:
        cand = solve_labeled(kp, w, h)
        err = compare(gt, cand, w, h)

    vis = draw_overlay(frame, cand, clicks) if cand is not None else frame.copy()
    s = TILE_W / w
    vis = cv2.resize(vis, (TILE_W, int(h * s)))
    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 68])
    src = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    if err is None:
        cap, cls = f"{len(kp)} keypoints &mdash; no pose", "bad"
    else:
        px, m = err["kp_px"], err["floor_m"]
        cls = "ok" if err["pass"] else ("near" if px < 60 else "bad")
        cap = (f'<b>{px:.0f} px</b> &middot; {m:.2f} m &middot; '
               f'{len(kp)} kp &middot; fits its own at {cand["err"]:.0f} px')
    return (f'<figure class="{cls}"><img src="{src}" alt="{img_path.stem}" loading="lazy">'
            f'<figcaption><span class="nm">{img_path.stem}</span>'
            f'<span class="note">{note}</span><span class="cap">{cap}</span>'
            f'</figcaption></figure>'), (err["kp_px"] if err else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?", default="reports/v9-fit.html")
    ap.add_argument("--model", default="finetune_out/court-pose-v9/weights/best.onnx")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    # ---- TEST: only frames whose ground truth passes the benchmark's gate ----
    test_tiles, test_errs = [], []
    for lp in sorted((ROOT / "data/test/labels").glob("*.json")):
        if len(test_tiles) >= a.n:
            break
        img = ROOT / "data/test" / (lp.stem + ".jpg")
        if not img.exists():
            continue
        clicks = {k: v for k, v in json.loads(lp.read_text()).items()
                  if not k.startswith("post_")}
        frame = cv2.imread(str(img))
        h, w = frame.shape[:2]
        if gt_reject(clicks, solve_labeled(clicks, w, h), frame):
            continue
        html, e = tile(img, clicks, a.model, "held out &mdash; never trained on")
        if html:
            test_tiles.append(html)
            if e is not None:
                test_errs.append(e)

    # ---- TRAIN: sampled evenly across the split so it is not all one venue ---
    train_imgs = sorted((ROOT / "data/dataset/court/images/train").glob("*.jpg"))
    step = max(1, len(train_imgs) // a.n)
    train_tiles, train_errs = [], []
    for img in train_imgs[::step][:a.n]:
        txt = ROOT / "data/dataset/court/labels/train" / (img.stem + ".txt")
        if not txt.exists():
            continue
        frame = cv2.imread(str(img))
        h, w = frame.shape[:2]
        html, e = tile(img, label_points(txt, w, h), a.model, "in the training set")
        if html:
            train_tiles.append(html)
            if e is not None:
                train_errs.append(e)

    def med(v):
        return float(np.median(v)) if v else float("nan")

    dest = ROOT / a.dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    model_name = Path(a.model).parent.parent.name
    dest.write_text(PAGE.format(
        model=model_name,
        n_train=len(train_tiles), n_test=len(test_tiles),
        med_train=med(train_errs), med_test=med(test_errs),
        train="\n".join(train_tiles), test="\n".join(test_tiles),
    ), encoding="utf-8")
    print(f"{dest}  ({dest.stat().st_size/1024/1024:.1f} MB)")
    print(f"  train {len(train_tiles)} frames, median {med(train_errs):.0f}px")
    print(f"  test  {len(test_tiles)} frames, median {med(test_errs):.0f}px")


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{model} — fit on train vs held-out test</title>
<style>
:root {{
  --ground:#eef1f2; --surface:#fff; --ink:#101a1d; --muted:#5f7178; --hair:#d3dbde;
  --ok:#00727e; --near:#b0762a; --bad:#b23b52;
  --mono: ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --ground:#0c1113; --surface:#141b1e; --ink:#e3ecef; --muted:#8ea0a7;
           --hair:#273338; --ok:#37c2ce; --near:#dda44f; --bad:#e8697f; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:system-ui,sans-serif; }}
.wrap {{ max-width:1320px; margin:0 auto; padding:36px 20px 80px; }}
h1 {{ font-family:var(--mono); font-size:clamp(22px,3.4vw,32px); letter-spacing:-.03em; margin:0; }}
h2 {{ font-family:var(--mono); font-size:18px; letter-spacing:-.02em; margin:0 0 4px; }}
p {{ max-width:70ch; line-height:1.6; }}
.sub {{ color:var(--muted); margin:10px 0 0; }}
section {{ margin-top:44px; }}
.head {{ display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
         border-bottom:1px solid var(--hair); padding-bottom:10px; margin-bottom:18px; }}
.stat {{ font-family:var(--mono); font-size:14px; color:var(--muted); }}
.stat b {{ color:var(--ink); font-size:17px; }}
.grid {{ display:grid; gap:18px; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); }}
figure {{ margin:0; background:var(--surface); border:1px solid var(--hair);
          border-radius:3px; overflow:hidden; }}
figure img {{ width:100%; display:block; }}
figcaption {{ padding:9px 11px; font-size:12.5px; display:flex; flex-direction:column; gap:2px; }}
.nm {{ font-family:var(--mono); font-weight:600; }}
.note {{ color:var(--muted); font-size:11.5px; }}
.cap {{ font-family:var(--mono); font-size:12px; color:var(--muted); }}
figure.ok {{ border-left:3px solid var(--ok); }}
figure.near {{ border-left:3px solid var(--near); }}
figure.bad {{ border-left:3px solid var(--bad); }}
figure.ok .cap b {{ color:var(--ok); }}
figure.bad .cap b {{ color:var(--bad); }}
</style>
<div class="wrap">
<h1>{model} &mdash; what it fits, and what it misses</h1>
<p class="sub">Cyan is the model&rsquo;s projected court, green crosses are the hand-clicked
points. Every caption reports error against the ground-truth <em>pose</em>, not against the
model&rsquo;s own detections &mdash; a pose can fit its own keypoints tightly and still sit
hundreds of pixels off the court, and the last number in each caption shows exactly that gap.
Left border: teal within 20&nbsp;px and 1&nbsp;m, amber under 60&nbsp;px, red beyond.</p>

<section>
  <div class="head"><h2>Training set</h2>
    <span class="stat">{n_train} frames &middot; median <b>{med_train:.0f} px</b></span></div>
  <div class="grid">{train}</div>
</section>

<section>
  <div class="head"><h2>Held-out test set</h2>
    <span class="stat">{n_test} frames &middot; median <b>{med_test:.0f} px</b></span></div>
  <div class="grid">{test}</div>
</section>
</div>
"""


if __name__ == "__main__":
    main()

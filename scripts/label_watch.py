"""Watch web-image labels and render solved-court overlays for the clicker.

Run alongside calib.html: python -m uv run python scripts/label_watch.py
For each out/webimgs/labels/<stem>.json (>=5 pts) whose overlay is missing or
stale, solve the camera from the clicks and write the wireframe overlay to
out/webimgs/overlays/<stem>.jpg with the reprojection error stamped on it.
"""
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.calibrate import draw_overlay, solve_web

WEB = Path(__file__).parent.parent / "out/webimgs"
OVL = WEB / "overlays"
OVL.mkdir(parents=True, exist_ok=True)

print(f"watching {WEB / 'labels'} ... ctrl-c to stop")
while True:
    for lp in sorted((WEB / "labels").glob("*.json")):
        dst = OVL / (lp.stem + ".jpg")
        if dst.exists() and dst.stat().st_mtime >= lp.stat().st_mtime:
            continue
        img_path = next((p for e in (".jpg", ".jpeg", ".png")
                         if (p := WEB / (lp.stem + e)).exists()), None)
        if img_path is None:
            continue
        try:
            ann = json.loads(lp.read_text())
        except (json.JSONDecodeError, OSError):  # mid-write
            continue
        ann = {k: v for k, v in ann.items() if not k.startswith("post_")}
        frame = cv2.imread(str(img_path))
        if frame is None or len(ann) < 5:
            continue
        h, w = frame.shape[:2]
        try:
            calib, net_h = solve_web(ann, w, h)
            out = draw_overlay(frame, calib, ann, net_h=net_h)
            e = calib["err"]
            color = (0, 200, 0) if e < 15 else (0, 165, 255) if e < 40 else (0, 0, 255)
            verdict = "good" if e < 15 else "check worst points" if e < 40 else "BAD - relabel"
            nh = f" | net ~{net_h:.2f}m" if net_h else ""
            cv2.putText(out, f"floor err {e:.1f}px - {verdict}{nh}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        except Exception as ex:
            out = frame.copy()
            cv2.putText(out, "solve failed - need >=5 good pts", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            print(f"{lp.stem}: solve failed ({ex})")
        cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{lp.stem}: overlay written")
    time.sleep(1)

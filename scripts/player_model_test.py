"""Compare person-detector configs on a 300-frame menlo slice.
Reports on-court count distribution per config (12 expected)."""
import json
from collections import Counter

import cv2
from ultralytics import YOLO

from farmclip.players import Tracker, to_scene_players

CONFIGS = [
    # (name, clip, calib, weights, imgsz, conf, max_miss, need, start)
    ("menlo_s1280_coast45", "out/clip.mp4", "out/calib.json",
     "yolo11s.pt", 1280, 0.2, 45, 12, 800),
    ("mikasa_n960_coast45", "out/mikasa/clip.mp4", "out/mikasa/calib.json",
     "yolo11n.pt", 960, 0.25, 90, 4, 6000),  # 60fps -> 1.5s = 90 frames
]
for name, clip, calib_p, weights, imgsz, conf, max_miss, need, start in CONFIGS:
    calib = json.load(open(calib_p))
    model = YOLO(weights)
    tracker = Tracker(max_miss=max_miss)
    cap = cv2.VideoCapture(clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    counts = Counter()
    for i in range(300):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, classes=[0], conf=conf, imgsz=imgsz, verbose=False)[0]
        tracked = tracker.update([b.xyxy[0].tolist() for b in r.boxes])
        counts[len(to_scene_players(tracked, calib))] += 1
    cap.release()
    ge = sum(v for k, v in counts.items() if k >= need)
    print(f"{name}: >={need} in {100*ge//300}% | dist {dict(sorted(counts.items()))}")

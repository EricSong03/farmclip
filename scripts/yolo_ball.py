"""YOLO ball bake-off contender: yolo11n, class 32 (sports ball), whole clip."""
import csv

import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # auto-downloads once
cap = cv2.VideoCapture("out/clip.mp4")
rows = []
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.predict(frame, classes=[32], conf=0.15, verbose=False)[0]
    if len(r.boxes):
        b = r.boxes[r.boxes.conf.argmax()]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        rows.append([i, 1, (x1 + x2) / 2, (y1 + y2) / 2, float(b.conf)])
    else:
        rows.append([i, 0, -1, -1, 0])
    i += 1
cap.release()
with open("out/ball_yolo.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Frame", "Visibility", "X", "Y", "Conf"])
    w.writerows(rows)
vis = sum(1 for r in rows if r[1])
print(f"{i} frames, {vis} detections ({100*vis//max(i,1)}%)")

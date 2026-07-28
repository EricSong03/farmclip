"""Step 0 go/no-go: can the teacher (yolo11x@1536) see far-side net-occluded players?

Renders teacher-box overlays on frames spread across the menlo clip so a human
(or Claude) can eyeball whether the hard class — far-side players behind the net
band — is detected. If the teacher misses them, distillation can't teach them and
the run must pivot to Roboflow data instead.
"""
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

VID = Path("videos/clip.mp4")
OUT = Path("out/debug/step0_teacher")
OUT.mkdir(parents=True, exist_ok=True)
IMGSZ = 1536
CONF = 0.25

model = YOLO("yolo11x.pt")
cap = cv2.VideoCapture(str(VID))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# spread across the clip
targets = [int(n * f) for f in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)]
for t in targets:
    cap.set(cv2.CAP_PROP_POS_FRAMES, t)
    ok, frame = cap.read()
    if not ok:
        continue
    r = model.predict(frame, classes=[0], conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    boxes = r.boxes.xyxy.tolist()
    confs = r.boxes.conf.tolist()
    vis = frame.copy()
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(vis, f"{c:.2f}", (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, f"frame {t}  n={len(boxes)}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(OUT / f"f{t:06d}.jpg"), vis)
    print(f"frame {t}: {len(boxes)} persons  confs={[round(c,2) for c in confs]}")
cap.release()
print(f"overlays -> {OUT}")

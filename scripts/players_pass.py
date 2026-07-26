"""Players v1 pass: YOLO persons -> tracks -> ground positions -> out/players.json.
Also renders verification overlays (boxes + reprojected court positions)."""
import json

import cv2
from ultralytics import YOLO

from farmclip.players import Tracker, to_scene_players

calib = json.load(open("out/calib.json"))
model = YOLO("yolo11n.pt")
tracker = Tracker()
cap = cv2.VideoCapture("out/clip.mp4")
per_frame = {}
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.predict(frame, classes=[0], conf=0.25, verbose=False)[0]
    boxes = [b.xyxy[0].tolist() for b in r.boxes]
    tracked = tracker.update(boxes)
    players = to_scene_players(tracked, calib)
    if players:
        per_frame[i] = players
    if i in (450, 900, 1350):  # verification frames
        vis = frame.copy()
        for tid, (x1, y1, x2, y2) in tracked:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(vis, str(tid), (int(x1), int(y1) - 4), 0, 0.6, (0, 255, 0), 2)
        for p in players:
            cv2.putText(vis, f"{p['id']}@({p['pos'][0]:.1f},{p['pos'][1]:.1f})",
                        (10, 30 + 18 * players.index(p)), 0, 0.5, (0, 255, 255), 1)
        cv2.imwrite(f"out/debug/players_{i:04d}.jpg", vis)
    i += 1
cap.release()
json.dump(per_frame, open("out/players.json", "w"))
counts = [len(v) for v in per_frame.values()]
print(f"{i} frames, {len(per_frame)} with players, "
      f"avg on-court {sum(counts)/max(len(counts),1):.1f}")

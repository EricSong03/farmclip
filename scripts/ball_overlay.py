"""Overlay ball CSV detections on sample frames with a 15-frame trail."""
import pandas as pd
import cv2

df = pd.read_csv("out/ball_vballnet/clip/ball.csv")
vis = df[df.Visibility > 0]
cap = cv2.VideoCapture("out/clip.mp4")
for target in [150, 450, 750, 1050, 1350, 1650]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok:
        continue
    trail = vis[(vis.Frame >= target - 15) & (vis.Frame <= target)]
    for _, r in trail.iterrows():
        age = (target - r.Frame) / 15
        cv2.circle(frame, (int(r.X), int(r.Y)), 6, (0, int(255 * (1 - age)), 255), 2)
    cur = vis[vis.Frame == target]
    for _, r in cur.iterrows():
        cv2.circle(frame, (int(r.X), int(r.Y)), 10, (0, 255, 0), 3)
    cv2.imwrite(f"out/debug/ball_{target:04d}.jpg", frame)
    print(target, "trail pts:", len(trail))
cap.release()

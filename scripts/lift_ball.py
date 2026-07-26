"""Lift the VballNet 2D ball CSV to 3D, render verification overlays."""
import json

import cv2
import numpy as np
import pandas as pd

from farmclip.ball3d import lift, _project

df = pd.read_csv("out/ball_vballnet/clip/ball.csv")
track = df[["Frame", "X", "Y", "Visibility"]].to_numpy(float)[:, [0, 1, 2, 3]]
calib = json.load(open("out/calib.json"))
ball3d, n_seg, n_ok = lift(track, calib, fps=30.0)
print(f"segments: {n_seg}, fitted: {n_ok}, frames with 3D ball: {len(ball3d)}")
json.dump(ball3d, open("out/ball3d.json", "w"))

if ball3d:
    ys = [p[1] for p in ball3d.values()]
    print("height range:", round(min(ys), 1), "-", round(max(ys), 1), "m")

# verification overlay: reprojected 3D path (cyan) vs raw detections (green)
frames_with = sorted(ball3d)
if frames_with:
    mid = frames_with[len(frames_with) // 2]
    seg_frames = [f for f in frames_with if abs(f - mid) < 45]
    cap = cv2.VideoCapture("out/clip.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, seg_frames[-1])
    _, frame = cap.read()
    cap.release()
    pts3d = np.array([ball3d[f] for f in seg_frames])
    proj = _project(calib, pts3d)
    for p in proj:
        cv2.circle(frame, tuple(p.astype(int)), 5, (255, 255, 0), 2)
    det = df[(df.Frame.isin(seg_frames)) & (df.Visibility > 0)]
    for _, r in det.iterrows():
        cv2.circle(frame, (int(r.X), int(r.Y)), 3, (0, 255, 0), -1)
    cv2.imwrite("out/debug/ball3d_reproj.jpg", frame)
    print("overlay: out/debug/ball3d_reproj.jpg around frame", mid)

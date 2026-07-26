import sys

import cv2

for i in map(int, sys.argv[1:]):
    cap = cv2.VideoCapture("out/clip.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    cap.release()
    p = f"out/debug/frame_{i:04d}.jpg"
    cv2.imwrite(p, frame)
    print(p)

"""Scan clip for frames with a visible far-court white line (horizontal
segment in the y 300-400 band, mid-image x range). Saves top candidates."""
import cv2
import numpy as np

from farmclip.lines import detect_segments, draw_segments
from farmclip.video import read_frames

best = []
for i, t, frame in read_frames("out/clip.mp4", step=30):
    segs = detect_segments(frame)
    score = 0.0
    hits = []
    for s in segs:
        x1, y1, x2, y2 = s
        ymid = (y1 + y2) / 2
        L = np.hypot(x2 - x1, y2 - y1)
        slope = abs(y2 - y1) / max(abs(x2 - x1), 1)
        if 300 < ymid < 410 and L > 120 and slope < 0.2 and 300 < (x1 + x2) / 2 < 1000:
            score += L
            hits.append(s)
    if score > 0:
        best.append((score, i, hits))

best.sort(reverse=True)
print("top frames:", [(i, round(s)) for s, i, _ in best[:6]])
for rank, (score, i, hits) in enumerate(best[:3]):
    cap = cv2.VideoCapture("out/clip.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    _, frame = cap.read()
    cap.release()
    segs = detect_segments(frame)
    out = draw_segments(frame, segs)
    cv2.imwrite(f"out/debug/farscan_{rank}_f{i}.jpg", out)
    import json
    json.dump([list(map(float, s)) for s in segs], open(f"out/segments_f{i}.json", "w"))
    print(f"saved farscan_{rank}_f{i}.jpg with {len(segs)} segs")

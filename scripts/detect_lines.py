import json
import sys

import cv2

from farmclip.lines import detect_segments, draw_segments

frame_path = sys.argv[1] if len(sys.argv) > 1 else "out/debug/frame_0600.jpg"
tag = sys.argv[2] if len(sys.argv) > 2 else "0600"
frame = cv2.imread(frame_path)
segs = detect_segments(frame)
print(len(segs), "segments")
json.dump([list(map(float, s)) for s in segs], open(f"out/segments_{tag}.json", "w"))
cv2.imwrite(f"out/debug/segments_{tag}.jpg", draw_segments(frame, segs))

"""crop.py FRAME OUTPREFIX x0 y0 x1 y1 [scale] — zoomed grid crop for visual annotation."""
import sys

import cv2

frame, prefix = sys.argv[1], sys.argv[2]
x0, y0, x1, y1 = map(int, sys.argv[3:7])
scale = int(sys.argv[7]) if len(sys.argv) > 7 else 3
img = cv2.imread(frame)
c = img[y0:y1, x0:x1]
c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
for gx in range(0, x1 - x0, 25):
    cv2.line(c, (gx * scale, 0), (gx * scale, c.shape[0]), (0, 255, 255), 1)
    cv2.putText(c, str(x0 + gx), (gx * scale + 2, 14), 0, 0.4, (0, 255, 255), 1)
for gy in range(0, y1 - y0, 25):
    cv2.line(c, (0, gy * scale), (c.shape[1], gy * scale), (0, 255, 255), 1)
    cv2.putText(c, str(y0 + gy), (2, gy * scale - 2), 0, 0.4, (0, 255, 255), 1)
out = f"out/debug/{prefix}.jpg"
cv2.imwrite(out, c)
print(out)

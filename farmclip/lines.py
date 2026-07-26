"""White court-line segment detection for calibration assistance.

Finds bright line segments on the floor; Claude assigns them to court-model
lines by reading the numbered debug overlay, then line intersections become
precise solvePnP correspondences.
"""

import cv2
import numpy as np


def detect_segments(frame, y_min=200):
    """Return merged white segments [(x1,y1,x2,y2), ...] below y_min."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # white lines = locally bright: tophat picks thin bright structures off wood
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
    _, mask = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
    mask[:y_min] = 0
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=40,
                           minLineLength=40, maxLineGap=8)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4)
    return _merge(segs)


def _merge(segs, angle_tol=np.deg2rad(3), dist_tol=12):
    """Greedy merge of nearly-collinear overlapping segments."""
    used = np.zeros(len(segs), bool)
    out = []
    for i in range(len(segs)):
        if used[i]:
            continue
        group = [segs[i]]
        used[i] = True
        a1 = np.arctan2(segs[i][3] - segs[i][1], segs[i][2] - segs[i][0])
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            a2 = np.arctan2(segs[j][3] - segs[j][1], segs[j][2] - segs[j][0])
            d = abs((a1 - a2 + np.pi / 2) % np.pi - np.pi / 2)
            if d > angle_tol:
                continue
            # midpoint distance to line i
            mx, my = (segs[j][0] + segs[j][2]) / 2, (segs[j][1] + segs[j][3]) / 2
            n = np.array([-(segs[i][3] - segs[i][1]), segs[i][2] - segs[i][0]], float)
            n /= np.linalg.norm(n)
            if abs(n @ [mx - segs[i][0], my - segs[i][1]]) < dist_tol:
                group.append(segs[j])
                used[j] = True
        pts = np.array(group).reshape(-1, 2)
        # fit line through all endpoints, take extremes along direction
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        t = (pts - [x0, y0]) @ [vx, vy]
        p1 = (x0 + t.min() * vx, y0 + t.min() * vy)
        p2 = (x0 + t.max() * vx, y0 + t.max() * vy)
        out.append((*p1, *p2))
    # longest first, cap for readability
    out.sort(key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
    return out[:30]


def draw_segments(frame, segs):
    out = frame.copy()
    for k, (x1, y1, x2, y2) in enumerate(segs):
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.putText(out, str(k), (mx + 3, my - 3), 0, 0.7, (0, 255, 0), 2)
    return out


def intersect(seg_a, seg_b):
    """Intersection point of the infinite lines through two segments."""
    x1, y1, x2, y2 = seg_a
    x3, y3, x4, y4 = seg_b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return (px, py)

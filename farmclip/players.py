"""Players v1: person bboxes -> feet point -> ground-plane position -> IoU tracks.

Ground position: back-project the bbox bottom-center through the calibrated
camera, intersect the ray with the floor (y=0).
"""

import numpy as np
import cv2


def floor_ray(calib, u, v):
    """Pixel -> world point on y=0 plane. Returns (x, z) or None if ray misses."""
    K = np.array([[calib["f"], 0, calib["img_w"] / 2],
                  [0, calib["f"], calib["img_h"] / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array(calib["rvec"], float))
    t = np.array(calib["tvec"], float).reshape(3, 1)
    C = (-R.T @ t).ravel()                       # camera center, world
    d = R.T @ np.linalg.inv(K) @ np.array([u, v, 1.0])  # ray dir, world
    if abs(d[1]) < 1e-9:
        return None
    s = -C[1] / d[1]
    if s <= 0:
        return None
    p = C + s * d
    return float(p[0]), float(p[2])


class Tracker:
    """Greedy IoU tracker. ponytail: no Kalman/appearance — upgrade if ID swaps hurt."""

    def __init__(self, iou_min=0.3, max_miss=15):
        self.iou_min, self.max_miss = iou_min, max_miss
        self.tracks = {}  # id -> {"box": [x1,y1,x2,y2], "miss": int}
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0

    def update(self, boxes):
        """boxes: list of [x1,y1,x2,y2]. Returns list of (track_id, box)."""
        assigned = {}
        free = set(self.tracks)
        for bi, box in sorted(
            ((bi, box) for bi, box in enumerate(boxes)),
            key=lambda p: -max((self._iou(p[1], self.tracks[t]["box"]) for t in free), default=0),
        ):
            best, best_iou = None, self.iou_min
            for tid in free:
                i = self._iou(box, self.tracks[tid]["box"])
                if i > best_iou:
                    best, best_iou = tid, i
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"box": box, "miss": 0}
            else:
                free.discard(best)
            self.tracks[best] = {"box": box, "miss": 0}
            assigned[best] = box
        for tid in list(self.tracks):
            if tid not in assigned:
                self.tracks[tid]["miss"] += 1
                if self.tracks[tid]["miss"] > self.max_miss:
                    del self.tracks[tid]
        return list(assigned.items())


def to_scene_players(tracked, calib, margin=1.5):
    """tracked: [(tid, box), ...] -> scene players list; on-court only."""
    from .court import HL, HW
    img_h = calib["img_h"]
    out = []
    for tid, (x1, y1, x2, y2) in tracked:
        if y2 >= img_h - 4 or x1 <= 4 or x2 >= calib["img_w"] - 4:
            continue  # box clipped by frame edge (foreground spectators) -> feet unreliable
        pos = floor_ray(calib, (x1 + x2) / 2, y2)
        if pos is None:
            continue
        x, z = pos
        if abs(x) > HL / 2 + margin or abs(z) > HW + margin:
            continue  # bench, refs, crowd
        team = "a" if x >= 0 else "b"
        out.append({"id": f"{team}{tid}", "team": team, "pos": [round(x, 3), round(z, 3)]})
    return out

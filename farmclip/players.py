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

    def __init__(self, iou_min=0.3, max_miss=15, min_hits=5):
        self.iou_min, self.max_miss, self.min_hits = iou_min, max_miss, min_hits
        self.tracks = {}  # id -> {"box": [x1,y1,x2,y2], "miss": int, "hits": int}
        self.next_id = 1

    @staticmethod
    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0

    def update(self, boxes):
        """boxes: list of [x1,y1,x2,y2]. Returns list of (track_id, box).

        Missed tracks coast on their last box for up to max_miss frames, so
        brief detector flickers don't break "all N players present"."""
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
                hits = 1
            else:
                free.discard(best)
                hits = self.tracks[best].get("hits", 0) + 1
            self.tracks[best] = {"box": box, "miss": 0, "hits": hits}
            assigned[best] = box
        for tid in list(self.tracks):
            if tid not in assigned:
                self.tracks[tid]["miss"] += 1
                if self.tracks[tid]["miss"] > self.max_miss:
                    del self.tracks[tid]
                elif self.tracks[tid].get("hits", 0) >= self.min_hits:
                    assigned[tid] = self.tracks[tid]["box"]  # established tracks coast
        return list(assigned.items())


def to_scene_players(tracked, calib, margin=1.0, per_side=6, hits=None):
    """tracked: [(tid, box), ...] -> scene players list; on-court only.
    per_side caps each team at roster size (6 for 6v6, 2 for 2v2): coasting
    ghosts and refs can't inflate counts past what volleyball allows."""
    from .court import COURT_L, COURT_W
    hl = calib.get("court_l", COURT_L) / 2
    hw = calib.get("court_w", COURT_W) / 2
    img_h = calib["img_h"]
    out = []
    for tid, (x1, y1, x2, y2) in tracked:
        if y2 >= img_h - 4 or x1 <= 4 or x2 >= calib["img_w"] - 4:
            continue  # box clipped by frame edge (foreground spectators) -> feet unreliable
        pos = floor_ray(calib, (x1 + x2) / 2, y2)
        if pos is None:
            continue
        x, z = pos
        if abs(x) > hl + margin or abs(z) > hw + margin:
            continue  # bench, refs, crowd
        team = "a" if x >= 0 else "b"
        out.append({"id": f"{team}{tid}", "team": team, "pos": [round(x, 3), round(z, 3)]})
    # dedupe: two tracks within 0.4m are the same person double-boxed
    dedup = []
    for p in sorted(out, key=lambda p: -(hits or {}).get(int(p["id"][1:]), 0)):
        if all((p["pos"][0] - q["pos"][0]) ** 2 + (p["pos"][1] - q["pos"][1]) ** 2 > 0.16
               for q in dedup):
            dedup.append(p)
    # roster cap per side, established tracks first
    capped = []
    for team in ("a", "b"):
        capped += [p for p in dedup if p["team"] == team][:per_side]
    return capped

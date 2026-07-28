"""Measure ball blob shape at anchor detections: streak vs round + diameter.

At each anchor, ellipse-fit the local blob. Round & sharp -> minor axis is
the apparent diameter (weak depth cue: Z = f * 0.21 / minor_px). Elongated
-> motion streak; long axis is the motion direction, minor axis still the
diameter. Quality in [0,1] gates how much any consumer should trust it.
"""
import cv2
import numpy as np

BALL_D = 0.21  # m


def measure_anchor(gray, x, y, crop=64):
    """Return (minor_px, major_px, angle_deg, quality) or None."""
    h, w = gray.shape
    x0, y0 = int(np.clip(x - crop // 2, 0, w - crop)), int(np.clip(y - crop // 2, 0, h - crop))
    c = gray[y0:y0 + crop, x0:x0 + crop].astype(np.float32)
    bg = np.median(c)
    dev = np.abs(c - bg)
    thr = max(dev.max() * 0.4, 12.0)
    mask = (dev > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = x - x0, y - y0
    best = None
    for cn in cnts:
        if len(cn) < 5:
            continue
        d = cv2.pointPolygonTest(cn, (float(cx), float(cy)), True)
        if d < -6:          # blob must contain/neighbor the anchor point
            continue
        if best is None or cv2.contourArea(cn) > cv2.contourArea(best):
            best = cn
    if best is None or cv2.contourArea(best) < 6:
        return None
    (ex, ey), (a1, a2), ang = cv2.fitEllipse(best)
    minor, major = sorted((a1, a2))
    if minor < 3 or major > crop * 0.9:
        return None
    # quality: sharp round blobs score high; tiny/edge-clipped ones low
    contrast = min(dev.max() / 60.0, 1.0)
    q = float(np.clip(contrast * min(minor / 6.0, 1.0), 0, 1))
    return minor, major, ang, q


def detect_streaks(video, frame_ranges, max_per_frame=3):
    """Classical spike detector: three-frame differencing inside the given
    frame ranges finds fast-moving elongated blobs (motion-blur streaks) that
    appearance CNNs miss entirely. Returns rows [frame, x, y, 1, w] ready to
    join a weighted track; downstream physics gates the junk (swinging arms
    also streak — only ballistic-coherent runs survive fitting)."""
    cap = cv2.VideoCapture(str(video))
    rows = []
    for f0, f1 in frame_ranges:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(f0) - 1, 0))
        ok, a = cap.read()
        ok2, b = cap.read()
        if not (ok and ok2):
            continue
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.int16)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.int16)
        for f in range(int(f0), int(f1) + 1):
            ok, c = cap.read()
            if not ok:
                break
            gc = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.int16)
            # motion present both into and out of frame f -> object AT f
            m = np.minimum(np.abs(gb - ga), np.abs(gc - gb)) > 16
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cands = []
            for cn in cnts:
                if not (40 < cv2.contourArea(cn) < 2500) or len(cn) < 5:
                    continue
                (ex, ey), (a1, a2), ang = cv2.fitEllipse(cn)
                minor, major = sorted((a1, a2))
                if major < 14 or minor > 26 or major / max(minor, 1) < 2.2:
                    continue
                cands.append((major / max(minor, 1), ex, ey))
            for _, ex, ey in sorted(cands, reverse=True)[:max_per_frame]:
                rows.append([f, ex, ey, 1.0, 0.3])
            ga, gb = gb, gc
    cap.release()
    return np.array(rows) if rows else np.empty((0, 5))


def measure_track(video, track, step=1):
    """{frame: (minor, major, angle, quality)} for visible anchors."""
    cap = cv2.VideoCapture(str(video))
    out = {}
    for row in track[track[:, 3] > 0][::step]:
        fr, x, y = int(row[0]), row[1], row[2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        m = measure_anchor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), x, y)
        if m is not None:
            out[fr] = m
    cap.release()
    return out

"""Lift 2D ball track to 3D via ballistic fits per flight segment.

Physics: between touches the ball is in free flight, X(t) = p0 + v0*t + g*t^2/2.
Fit (p0, v0) per segment by minimizing reprojection error against the 2D
detections through the calibrated camera. One camera + physics = full 3D.

ponytail: uses the single reference calib for all frames — the camera drifts
slightly (see known-issues); per-frame pose propagation upgrades this later.
"""

import numpy as np
import cv2
from scipy.optimize import least_squares

G = np.array([0.0, -9.81, 0.0])


def segments(track, min_len=8, max_gap=3, accel_px=25.0):
    """track: (N,4) [frame, x, y, vis]. Yields index arrays of flight segments,
    split at visibility gaps and 2D acceleration spikes (touches)."""
    vis = track[track[:, 3] > 0]
    if len(vis) < min_len:
        return
    cur = [0]
    for i in range(1, len(vis)):
        gap = vis[i, 0] - vis[i - 1, 0]
        split = gap > max_gap
        if not split and len(cur) >= 2:
            a, b, c = vis[cur[-2]], vis[cur[-1]], vis[i]
            # ponytail: touch = 2D accel spike; misses gentle sets along view axis
            accel = np.linalg.norm((c[1:3] - b[1:3]) - (b[1:3] - a[1:3]))
            split = accel > accel_px
        if split:
            if len(cur) >= min_len:
                yield vis[cur]
            cur = [i]
        else:
            cur.append(i)
    if len(cur) >= min_len:
        yield vis[cur]


def _project(calib, pts3d):
    K = np.array([[calib["f"], 0, calib["img_w"] / 2],
                  [0, calib["f"], calib["img_h"] / 2], [0, 0, 1]])
    proj, _ = cv2.projectPoints(np.asarray(pts3d, float),
                                np.array(calib["rvec"], float),
                                np.array(calib["tvec"], float), K, None)
    return proj.reshape(-1, 2)


def fit_segment(seg, calib, fps):
    """seg: (M,4) visible samples. Returns (frames, pts3d, rms_px) or None."""
    t = (seg[:, 0] - seg[0, 0]) / fps
    uv = seg[:, 1:3]

    def model(params):
        p0, v0 = params[:3], params[3:]
        return p0[None] + v0[None] * t[:, None] + 0.5 * G[None] * t[:, None] ** 2

    def residuals(params):
        return (_project(calib, model(params)) - uv).ravel()

    # init: mid-court at 3m, no velocity — LM sorts it out
    x0 = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    out = least_squares(residuals, x0, method="lm", max_nfev=2000)
    pts = model(out.x)
    rms = float(np.sqrt(np.mean(np.sum((_project(calib, pts) - uv) ** 2, axis=1))))
    # sanity: inside gym volume, above floor, humanly-possible speed
    speed = np.linalg.norm(out.x[3:] + G * t[-1])  # max of |v0|, |v(t_end)| below
    v0 = np.linalg.norm(out.x[3:])
    if rms > 15 or pts[:, 1].min() < -0.5 or pts[:, 1].max() > 12 \
            or np.abs(pts[:, 0]).max() > 15 or np.abs(pts[:, 2]).max() > 10 \
            or max(v0, speed) > 40:  # ~144 km/h; elite spikes top out ~37 m/s
        return None
    return seg[:, 0].astype(int), pts, rms


def lift(track, calib, fps):
    """Full pipeline: 2D track -> {frame: [x,y,z]}."""
    out = {}
    n_seg = n_ok = 0
    for seg in segments(track):
        n_seg += 1
        fit = fit_segment(seg, calib, fps)
        if fit is None:
            continue
        n_ok += 1
        frames, pts, rms = fit
        for fr, p in zip(frames, pts):
            out[int(fr)] = p.tolist()
    return out, n_seg, n_ok

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


def reject_outliers(track, win=5, tol_px=40.0):
    """Kill detections far from the local median of their neighbors (false
    positives on players/logos poison the ballistic fitter's windows)."""
    t = track.copy()
    vis_idx = np.where(t[:, 3] > 0)[0]
    for k, i in enumerate(vis_idx):
        lo, hi = max(0, k - win), min(len(vis_idx), k + win + 1)
        nb = t[vis_idx[lo:hi], 1:3]
        if len(nb) < 3:
            continue
        med = np.median(nb, axis=0)
        if np.linalg.norm(t[i, 1:3] - med) > tol_px:
            t[i, 3] = 0
    return t


def fill_gaps(track, max_gap=3):
    """Interpolate 2D detections across visibility gaps <= max_gap frames.
    track: (N,4) [frame, x, y, vis], one row per frame. Returns a copy."""
    t = track.copy()
    vis_idx = np.where(t[:, 3] > 0)[0]
    for a, b in zip(vis_idx, vis_idx[1:]):
        gap = b - a
        if 1 < gap <= max_gap:
            for k in range(1, gap):
                u = k / gap
                t[a + k, 1:3] = t[a, 1:3] * (1 - u) + t[b, 1:3] * u
                t[a + k, 3] = 1
    return t


def runs(track, max_gap=3):
    """Contiguous visible runs, split only at visibility gaps > max_gap."""
    vis = track[track[:, 3] > 0]
    if not len(vis):
        return
    cur = [0]
    for i in range(1, len(vis)):
        if vis[i, 0] - vis[i - 1, 0] > max_gap:
            yield vis[cur]
            cur = [i]
        else:
            cur.append(i)
    yield vis[cur]


def grow_segments(run, calib, fps, min_len=6, rms_ok=12.0):
    """Greedy fit-quality segmentation: extend the window while a ballistic fit
    explains it; where the fit breaks is a touch. Yields fitted segments."""
    i = 0
    n = len(run)
    while i + min_len <= n:
        j = i + min_len
        best = fit_segment(run[i:j], calib, fps, rms_ok)
        if best is None:
            i += 1  # can't even fit the seed window; slide past the junk
            continue
        while j < n:
            j2 = min(j + 4, n)
            trial = fit_segment(run[i:j2], calib, fps, rms_ok)
            if trial is None:
                break
            best, j = trial, j2
        yield best
        i = j


def _project(calib, pts3d):
    K = np.array([[calib["f"], 0, calib["img_w"] / 2],
                  [0, calib["f"], calib["img_h"] / 2], [0, 0, 1]])
    proj, _ = cv2.projectPoints(np.asarray(pts3d, float),
                                np.array(calib["rvec"], float),
                                np.array(calib["tvec"], float), K, None)
    return proj.reshape(-1, 2)


def fit_segment(seg, calib, fps, rms_ok=15.0):
    """seg: (M,4) visible samples. Returns (frames, pts3d, rms_px, params) or None."""
    t = (seg[:, 0] - seg[0, 0]) / fps
    uv = seg[:, 1:3]

    def model(params):
        p0, v0 = params[:3], params[3:]
        return p0[None] + v0[None] * t[:, None] + 0.5 * G[None] * t[:, None] ** 2

    def residuals(params):
        return (_project(calib, model(params)) - uv).ravel()

    # bounded fit: constrain p0 to the gym volume and v0 to humanly-possible
    # speeds, so the optimizer finds the best PHYSICAL trajectory rather than
    # escaping to infinity and getting rejected afterwards
    x0 = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    out = least_squares(residuals, x0, method="trf", max_nfev=2000,
                        bounds=([-12, 0.0, -8, -45, -45, -45],
                                [12, 10.0, 8, 45, 45, 45]))
    pts = model(out.x)
    rms = float(np.sqrt(np.mean(np.sum((_project(calib, pts) - uv) ** 2, axis=1))))
    if rms > rms_ok or pts[:, 1].max() > 12 \
            or np.abs(pts[:, 0]).max() > 15 or np.abs(pts[:, 2]).max() > 10:
        return None
    return seg[:, 0].astype(int), pts, rms, out.x


def lift(track, calib, fps):
    """Full pipeline: 2D track -> {frame: [x,y,z]}.

    Emits EVERY frame across a fitted segment's span (dense), not just
    detected ones — the ballistic model interpolates interior gaps."""
    fits = []
    for run in runs(track):
        fits.extend(grow_segments(run, calib, fps))
    n_seg = len(fits)

    # continuity filter: the ball's POSITION is continuous through touches.
    # A junction jumping faster than 25 m/s means one side is junk - drop the
    # shorter segment and re-check.
    def span(fit):
        return int(fit[0][0]), int(fit[0][-1])

    def pos_at(fit, fr):
        p0, v0 = fit[3][:3], fit[3][3:]
        t = (fr - fit[0][0]) / fps
        return p0 + v0 * t + 0.5 * G * t * t

    fits.sort(key=lambda f: f[0][0])
    changed = True
    while changed:
        changed = False
        for i in range(len(fits) - 1):
            a, b = fits[i], fits[i + 1]
            gap = span(b)[0] - span(a)[1]
            if not (0 < gap <= fps * 0.7):
                continue
            jump = np.linalg.norm(pos_at(b, span(b)[0]) - pos_at(a, span(a)[1]))
            # allow spike-speed travel plus ~1m calibration slack; only cull
            # when the offender is also short (junk signature)
            if jump > 1.0 + 35 * gap / fps:
                short_i = i if len(a[0]) < len(b[0]) else i + 1
                if len(fits[short_i][0]) < 10:
                    del fits[short_i]
                    changed = True
                    break
    n_ok = len(fits)

    out = {}
    for fit in fits:
        frames, pts, rms, params = fit
        f0, f1 = int(frames[0]), int(frames[-1])
        p0, v0 = params[:3], params[3:]
        for fr in range(f0, f1 + 1):
            t = (fr - f0) / fps
            out[fr] = (p0 + v0 * t + 0.5 * G * t * t).tolist()
    # bridge short inter-segment gaps (the touch moment): linear between the
    # adjacent flight endpoints — the ball is at/near the touching player
    frames_sorted = sorted(out)
    for a, b in zip(frames_sorted, frames_sorted[1:]):
        gap = b - a
        if 1 < gap <= int(fps * 0.7):
            pa, pb = np.array(out[a]), np.array(out[b])
            if np.linalg.norm(pb - pa) > 1.0 + 35 * gap / fps:
                continue  # endpoints too far apart: not the same touch, don't teleport
            for k in range(1, gap):
                u = k / gap
                out[a + k] = (pa * (1 - u) + pb * u).tolist()
    return out, n_seg, n_ok

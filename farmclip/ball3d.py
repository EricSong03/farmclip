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


def consensus(a, b, agree_px=12.0):
    """Merge two model tracks keeping only frames where both agree.
    a, b: (N,4) [frame,x,y,vis]. Physics-first: these are the ONLY anchors;
    everything else is reconstructed, never guessed."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    t = a.copy()
    both = (a[:, 3] > 0) & (b[:, 3] > 0)
    close = np.linalg.norm(a[:, 1:3] - b[:, 1:3], axis=1) <= agree_px
    keep = both & close
    t[:, 1:3] = (a[:, 1:3] + b[:, 1:3]) / 2
    t[:, 3] = keep.astype(float)
    t[~keep, 1:3] = 0
    return t


def reject_static(track, fps, win_s=0.5, min_px=6.0):
    """Kill anchors that barely move across a ~win_s window. A ball in free
    flight ALWAYS moves (even at apex gravity drops it >10px in 0.25s);
    consensus false positives on scoreboards/banners/logos don't."""
    t = track.copy()
    vis_idx = np.where(t[:, 3] > 0)[0]
    win_f = win_s * fps
    for i in vis_idx:
        nb = vis_idx[(np.abs(t[vis_idx, 0] - t[i, 0]) <= win_f)]
        far = t[nb, 0].max() - t[nb, 0].min() >= win_f * 0.6
        if len(nb) >= 3 and far and \
                np.linalg.norm(t[nb, 1:3] - t[i, 1:3], axis=1).max() < min_px:
            t[i, 3] = 0
    return t


def velocity_segments(run, calib, fps, soft=2.5, hard=7.0, min_anchors=4):
    """Split an anchor run at velocity breaks (= contacts). Yields
    (anchors, soft_entry) — soft_entry True when the cut before this segment
    was a slight touch, so the fit may seed from the previous arc's exit.

    Threshold logic: between anchors the only allowed acceleration is
    gravity (+drag, absorbed in the soft factor). Residual acceleration
    beyond `soft`x gravity ends the arc; beyond `hard`x it's a real hit.
    ponytail: gravity in px assumes ~8m depth; refit per-arc later if it
    proves too coarse."""
    fr, p = run[:, 0], run[:, 1:3]
    if len(run) < 2 * min_anchors:
        if len(run) >= min_anchors:
            yield run, False
        return
    dt = np.diff(fr) / fps
    v = np.diff(p, axis=0) / dt[:, None]                # px/s between anchors
    g_px = calib["f"] * 9.81 / 8.0                      # px/s^2 at mid-court
    cut_at, cut_hard = [], []
    for i in range(1, len(v)):
        dtm = (dt[i - 1] + dt[i]) / 2
        resid = np.linalg.norm(v[i] - v[i - 1] - [0, g_px * dtm]) / dtm
        if resid > soft * g_px:
            cut_at.append(i)                            # cut before anchor i+? -> at point i
            cut_hard.append(resid > hard * g_px)
    lo, soft_entry = 0, False
    for c, is_hard in zip(cut_at, cut_hard):
        if c + 1 - lo >= min_anchors:
            yield run[lo:c + 1], soft_entry
            lo, soft_entry = c + 1, not is_hard
        elif is_hard:
            lo, soft_entry = c + 1, False               # too short to fit; drop it
    if len(run) - lo >= min_anchors:
        yield run[lo:], soft_entry


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


def fit_segment(seg, calib, fps, rms_ok=15.0, x0=None):
    """seg: (M,4) visible samples. Returns (frames, pts3d, rms_px, params) or None.
    x0 optionally seeds the optimizer (soft-break continuity from prior arc)."""
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
    if x0 is None:
        x0 = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    x0 = np.clip(x0, [-12, 0.0, -8, -45, -45, -45], [12, 10.0, 8, 45, 45, 45])
    # robust loss: single-frame detector glitches (2nd union model firing on
    # another object) must not poison an otherwise-clean arc
    out = least_squares(residuals, x0, method="trf", max_nfev=2000,
                        loss="soft_l1", f_scale=6.0,
                        bounds=([-12, 0.0, -8, -45, -45, -45],
                                [12, 10.0, 8, 45, 45, 45]))
    pts = model(out.x)
    d = np.linalg.norm(_project(calib, pts) - uv, axis=1)
    inliers = d < 20
    if inliers.mean() < 0.7:
        return None
    rms = float(np.sqrt(np.mean(d[inliers] ** 2)))
    if rms > rms_ok or pts[:, 1].max() > 12 \
            or np.abs(pts[:, 0]).max() > 15 or np.abs(pts[:, 2]).max() > 10:
        return None
    return seg[:, 0].astype(int), pts, rms, out.x


def lift(track, calib, fps, min_len=5, rms_ok=14.0, segmenter="greedy"):
    """Full pipeline: 2D track -> {frame: [x,y,z]}.

    Emits EVERY frame across a fitted segment's span (dense), not just
    detected ones — the ballistic model interpolates interior gaps.
    segmenter: "greedy" (fit-residual growing, legacy) or "velocity"
    (physics-first: cut at velocity breaks, seed across soft touches)."""
    fits = []
    if segmenter == "velocity":
        # consensus anchors are sparse but trusted: let arcs span long
        # detection gaps — the velocity-break test still guards contacts
        track = reject_static(track, fps)
        pairs = []  # (seg, fit) so merges can refit on raw anchors
        for run in runs(track, max_gap=int(fps * 0.5)):
            prev = None
            for seg, soft_entry in velocity_segments(run, calib, fps,
                                                     min_anchors=min_len - 1):
                x0 = None
                if soft_entry and prev is not None:
                    p0, v0 = prev[3][:3], prev[3][3:]
                    te = (seg[0, 0] - prev[0][0]) / fps
                    x0 = np.concatenate([p0 + v0 * te + 0.5 * G * te * te,
                                         v0 + G * te])
                fit = fit_segment(seg, calib, fps, rms_ok, x0=x0)
                if fit is not None:
                    pairs.append((seg, fit))
                prev = fit
        # rescue pass: velocity breaks over-cut on noise, leaving dense anchor
        # clusters as sub-min fragments that never got fit. Sweep uncovered
        # anchors with the greedy fit-grower — physics still gates acceptance.
        used = set()
        for sa, _ in pairs:
            used.update(sa[:, 0].astype(int))
        t2 = track.copy()
        t2[np.isin(t2[:, 0].astype(int), list(used)), 3] = 0
        for run in runs(t2, max_gap=int(fps * 0.5)):
            for fit in grow_segments(run, calib, fps, min_len=min_len,
                                     rms_ok=rms_ok):
                seg = run[np.isin(run[:, 0], fit[0])]
                pairs.append((seg, fit))
        # leftover anchors too few to fit alone ride along as fit-less
        # fragments; the merge pass below can absorb them into real arcs
        used = {int(f) for sa, _ in pairs for f in sa[:, 0]}
        t3 = track.copy()
        t3[np.isin(t3[:, 0].astype(int), list(used)), 3] = 0
        for run in runs(t3, max_gap=int(fps * 0.5)):
            pairs.append((run, None))
        pairs.sort(key=lambda sf: sf[0][0, 0])
        # merge pass: velocity breaks over-cut on anchor noise. If ONE
        # ballistic fit explains two adjacent arcs' anchors, they were the
        # same flight — physics decides, not the threshold.
        merged = True
        while merged:
            merged = False
            for i in range(len(pairs) - 1):
                (sa, fa), (sb, fb) = pairs[i], pairs[i + 1]
                if fa is None and fb is None:
                    continue
                if sb[0, 0] - sa[-1, 0] > fps * 0.5:
                    continue
                both = np.vstack([sa, sb])
                trial = fit_segment(both, calib, fps, rms_ok,
                                    x0=(fa or fb)[3])
                if trial is not None:
                    pairs[i:i + 2] = [(both, trial)]
                    merged = True
                    break
        fits = [f for _, f in pairs if f is not None]
    else:
        for run in runs(track):
            fits.extend(grow_segments(run, calib, fps, min_len=min_len, rms_ok=rms_ok))
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
    # adjacent flight endpoints — the ball is at/near the touching player.
    # A touch is split-second and local: long/far bridges drew straight
    # cross-gym "trajectories", so both limits are tight now.
    frames_sorted = sorted(out)
    for a, b in zip(frames_sorted, frames_sorted[1:]):
        gap = b - a
        if 1 < gap <= int(fps * 0.25):
            pa, pb = np.array(out[a]), np.array(out[b])
            if np.linalg.norm(pb - pa) > 2.5:
                continue  # endpoints too far apart: not the same touch, don't teleport
            for k in range(1, gap):
                u = k / gap
                out[a + k] = (pa * (1 - u) + pb * u).tolist()
    # teleport smoother: adjacent-segment junctions can still jump within one
    # frame; crossfade a +-3-frame window over any superphysical step
    frames_sorted = sorted(out)
    for a, b in zip(frames_sorted, frames_sorted[1:]):
        if b - a != 1:
            continue
        if np.linalg.norm(np.array(out[b]) - np.array(out[a])) * fps > 45:
            lo = next((f for f in range(a - 3, a + 1) if f in out), a)
            hi = next((f for f in range(b + 3, b - 1, -1) if f in out), b)
            pl, ph = np.array(out[lo]), np.array(out[hi])
            for fr in range(lo, hi + 1):
                u = (fr - lo) / max(hi - lo, 1)
                out[fr] = (pl * (1 - u) + ph * u).tolist()
    return out, n_seg, n_ok

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


def weighted_track(a, b, agree_px=12.0, w_single=0.35):
    """Merge two model tracks into (N,5) [frame,x,y,vis,w].
    Both agree within agree_px -> w=1 (position averaged); exactly one model
    fires -> w=w_single (its position, trusted only as far as physics
    agrees); else invisible. Physics-first amendment (2026-07-28, user call
    after watching overlays): single-model detections carry real signal in
    blur/far-court stretches — feed them to the fit at low weight instead
    of discarding them; the robust loss handles the junk."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    t = np.zeros((n, 5))
    t[:, 0] = a[:, 0]
    va, vb = a[:, 3] > 0, b[:, 3] > 0
    close = np.linalg.norm(a[:, 1:3] - b[:, 1:3], axis=1) <= agree_px
    both = va & vb & close
    t[both, 1:3] = (a[both, 1:3] + b[both, 1:3]) / 2
    t[both, 3], t[both, 4] = 1, 1.0
    only = (va | vb) & ~both
    src = np.where(va[:, None], a[:, 1:3], b[:, 1:3])
    t[only, 1:3] = src[only]
    t[only, 3], t[only, 4] = 1, w_single
    return t


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
    """seg: (M,4) or (M,5 with weight col) visible samples.
    Returns (frames, pts3d, rms_px, params) or None.
    x0 optionally seeds the optimizer (soft-break continuity from prior arc).
    Low-weight rows (single-model detections) pull the fit but never vote
    on the acceptance gates — physics arbitrates what they're worth."""
    t = (seg[:, 0] - seg[0, 0]) / fps
    uv = seg[:, 1:3]
    wts = seg[:, 4] if seg.shape[1] > 4 else np.ones(len(seg))
    hi = wts >= 0.99
    lo_only = not hi.any()
    if lo_only:
        # no consensus anchor at all: only accept if MANY single-model
        # detections cohere on one ballistic path — junk can't fake that
        if len(seg) < 8:
            return None
        hi = np.ones(len(seg), bool)

    def model(params):
        p0, v0 = params[:3], params[3:]
        return p0[None] + v0[None] * t[:, None] + 0.5 * G[None] * t[:, None] ** 2

    def residuals(params):
        return ((_project(calib, model(params)) - uv) * wts[:, None]).ravel()

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
    d_gate, frac, rms_cap = (12, 0.8, min(rms_ok, 8.0)) if lo_only \
        else (20, 0.7, rms_ok)
    inliers = (d < d_gate) & hi
    if inliers.sum() / hi.sum() < frac:
        return None
    rms = float(np.sqrt(np.mean(d[inliers] ** 2)))
    rms_ok = rms_cap
    if rms > rms_ok or pts[:, 1].max() > 12 \
            or np.abs(pts[:, 0]).max() > 15 or np.abs(pts[:, 2]).max() > 10:
        return None
    return seg[:, 0].astype(int), pts, rms, out.x


def measure_net_bands(video, calib, fps, every_s=30.0):
    """Detect the net's top band in the actual pixels every ~every_s and
    stash the segments on the calib dict. Trigger lines built from real
    pixels are immune to both calibration bias (mikasa projects the net
    ~50px high, err 16.7px) and camera drift over long videos."""
    from .lines import detect_segments
    h, w = calib["img_h"], calib["img_w"]
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bands = []
    for fr in range(0, n, int(every_s * fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            continue
        segs = detect_segments(img)
        cand = [s for s in segs
                if abs(s[3] - s[1]) / max(abs(s[2] - s[0]), 1) < 0.1
                and h * 0.2 < (s[1] + s[3]) / 2 < h * 0.6
                and abs(s[2] - s[0]) > w * 0.45]

        def ymid(s):
            return (s[1] + s[3]) / 2
        paired = [s for s in cand
                  if any(20 < ymid(o) - ymid(s) < 80 for o in cand if o is not s)]
        pool = paired or cand
        if pool:
            bands.append((fr, min(pool, key=ymid)))
    cap.release()
    if bands:
        calib["_net_bands"] = bands
    return len(bands)


def _net_mid(calib, frame=None):
    """Image-space net trigger line (x_img -> y_img). Prefers the measured
    band nearest `frame`; falls back to the projected net midline."""
    if frame is not None and calib.get("_net_bands"):
        fr, s = min(calib["_net_bands"], key=lambda b: abs(b[0] - frame))
        (x0, y0, x1, y1) = s[:4]
        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        return np.array([x0, x1], float), np.array([y0, y1], float)
    if "_netuv" not in calib:
        m = _project(calib, [[0.0, 1.7, z] for z in np.linspace(-4.5, 4.5, 50)])
        o = np.argsort(m[:, 0])
        calib["_netuv"] = (m[o, 0], m[o, 1])
    return calib["_netuv"]


def _backproject_at_dist(calib, u, v, dist):
    """3D point along the pixel ray at a given distance from the camera."""
    K = np.array([[calib["f"], 0, calib["img_w"] / 2],
                  [0, calib["f"], calib["img_h"] / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array(calib["rvec"], float))
    tvec = np.array(calib["tvec"], float).ravel()
    C = -R.T @ tvec
    d = R.T @ np.linalg.solve(K, np.array([u, v, 1.0]))
    return C + dist * d / np.linalg.norm(d)


def _ray_floor(calib, u, v):
    """Backproject pixel (u,v) and intersect with the floor plane y=0."""
    K = np.array([[calib["f"], 0, calib["img_w"] / 2],
                  [0, calib["f"], calib["img_h"] / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array(calib["rvec"], float))
    tvec = np.array(calib["tvec"], float).ravel()
    C = -R.T @ tvec
    d = R.T @ np.linalg.solve(K, np.array([u, v, 1.0]))
    if abs(d[1]) < 1e-9:
        return None
    s = -C[1] / d[1]
    return C + s * d if s > 0 else None


def refine_chains(pairs, calib, fps, max_gap_s=2.5):
    """Jointly refit each rally chain (arcs linked by gaps <= max_gap_s) with
    ball-position continuity across touches. Depth is under-determined per
    arc; continuity lets well-constrained arcs pull their neighbors' depth
    to the right place. Accepts a chain's joint solution only if summed
    reprojection stays within tolerance (pixels are the arbiter).
    Mutates and returns pairs."""
    W_C = 20.0  # px-equivalent penalty per meter of touch discontinuity

    def chain_ranges():
        out, s = [], 0
        for i in range(1, len(pairs)):
            if pairs[i][0][0, 0] - pairs[i - 1][0][-1, 0] > fps * max_gap_s:
                out.append((s, i)); s = i
        out.append((s, len(pairs)))
        return out

    for lo, hi in chain_ranges():
        n = hi - lo
        if n < 2:
            continue
        segs = [pairs[i][0] for i in range(lo, hi)]
        f0s = [s[0, 0] for s in segs]
        uvs = [s[:, 1:3] for s in segs]
        ts = [(s[:, 0] - s[0, 0]) / fps for s in segs]
        wls = [s[:, 4] if s.shape[1] > 4 else np.ones(len(s)) for s in segs]
        mids = [(segs[k][-1, 0] + segs[k + 1][0, 0]) / 2 for k in range(n - 1)]

        def pos(P, k, tt):
            p0, v0 = P[6 * k:6 * k + 3], P[6 * k + 3:6 * k + 6]
            tt = np.atleast_1d(tt)
            return p0[None] + v0[None] * tt[:, None] + 0.5 * G[None] * tt[:, None] ** 2

        P0 = np.concatenate([pairs[i][1][3] for i in range(lo, hi)])

        # net-plane anchors: band-crossing times of the chain's dense
        # projected path. Inside the chain, pooled pixels + continuity make
        # occasional sweep-misclassifications survivable (soft_l1 tail).
        W_N = 15.0

        def net_events(P):
            ev = []  # (arc index k, local time tc)
            for k in range(n):
                nx_, ny_ = _net_mid(calib, frame=f0s[k])
                span = segs[k][-1, 0] - f0s[k]
                td = np.arange(0, span + 1) / fps
                duv = _project(calib, pos(P, k, td))
                dinb = (duv[:, 0] >= nx_[0]) & (duv[:, 0] <= nx_[-1])
                drel = duv[:, 1] - np.interp(duv[:, 0], nx_, ny_)
                for j in range(1, len(td)):
                    if dinb[j] and dinb[j - 1] and drel[j - 1] * drel[j] < 0:
                        u = drel[j - 1] / (drel[j - 1] - drel[j])
                        ev.append((k, td[j - 1] + u / fps))
            return ev

        net_ev = net_events(P0)

        # floor-bounce anchors: junction with a down->up reversal in image y
        # and a short gap = the ball hit the floor there; pixel ray ∩ floor
        # plane is an EXACT 3D point — the strongest depth anchor we have
        W_B = 20.0
        bounce_ev = []  # (junction index k, 3d point)
        for k in range(n - 1):
            gap_s = (segs[k + 1][0, 0] - segs[k][-1, 0]) / fps
            va = segs[k][-1, 2] - segs[k][-2, 2]        # image y velocity out
            vb = segs[k + 1][1, 2] - segs[k + 1][0, 2]  # image y velocity in
            if gap_s <= 0.6 and va > 0 and vb < 0:
                u = (segs[k][-1, 1] + segs[k + 1][0, 1]) / 2
                v = (segs[k][-1, 2] + segs[k + 1][0, 2]) / 2
                q = _ray_floor(calib, u, v)
                if q is not None and abs(q[0]) < 12 and abs(q[2]) < 9:
                    bounce_ev.append((k, q))

        def resid(P):
            r = [((_project(calib, pos(P, k, ts[k])) - uvs[k])
                  * wls[k][:, None]).ravel()
                 for k in range(n)]
            for k, fm in enumerate(mids):
                a = pos(P, k, (fm - f0s[k]) / fps)[0]
                b = pos(P, k + 1, (fm - f0s[k + 1]) / fps)[0]
                r.append(W_C * (a - b))
            r.append(np.array([W_N * (P[6 * k] + P[6 * k + 3] * tc)
                               for k, tc in net_ev]))
            for k, q in bounce_ev:
                fm = mids[k]
                a = pos(P, k, (fm - f0s[k]) / fps)[0]
                r.append(W_B * (a - q))
            return np.concatenate([x for x in r if len(x)])
        bl = np.tile([-12, 0.0, -8, -45, -45, -45], n)
        bu = np.tile([12, 10.0, 8, 45, 45, 45], n)
        sol = least_squares(resid, np.clip(P0, bl, bu), method="trf",
                            max_nfev=4000, loss="soft_l1", f_scale=6.0,
                            bounds=(bl, bu))
        # second pass: crossing times were detected on the seed; the refined
        # chain crosses at slightly different times — re-detect and re-solve
        # to pull in the late/early-crossing tail
        ev2 = net_events(sol.x)
        if ev2 != net_ev:
            net_ev = ev2
            sol = least_squares(resid, sol.x, method="trf",
                                max_nfev=2000, loss="soft_l1", f_scale=6.0,
                                bounds=(bl, bu))

        def rms_of(P):
            d = [np.minimum(np.linalg.norm(
                _project(calib, pos(P, k, ts[k])) - uvs[k], axis=1),
                20)[wls[k] >= 0.99]
                for k in range(n)]
            return np.sqrt(np.mean(np.concatenate(d) ** 2))

        if rms_of(sol.x) <= rms_of(np.clip(P0, bl, bu)) * 1.05 + 0.5:
            for k in range(n):
                i = lo + k
                frames, _, rms, _ = pairs[i][1]
                params = sol.x[6 * k:6 * k + 6]
                pts = pos(sol.x, k, ts[k])
                pairs[i] = (pairs[i][0], (frames, pts, rms, params))
    return pairs


def lift_with_streaks(track, calib, fps, video, max_gap_s=4.0):
    """Two-pass lift: fit once, run the classical streak detector inside the
    remaining dark gaps (spike legs where BOTH CNNs go blind), refit with the
    streak rows as low-weight observations. Physics gates their junk."""
    from .streaks import detect_streaks
    out, n_seg, n_ok, filled = lift(track, calib, fps, segmenter="velocity")
    frames = sorted(out)
    gaps = [(a, b) for a, b in zip(frames, frames[1:])
            if 1 < b - a <= fps * max_gap_s]
    if not gaps:
        return out, n_seg, n_ok, filled
    srows = detect_streaks(video, [(a + 1, b - 1) for a, b in gaps])
    for a, b in gaps:
        cand = srows[(srows[:, 0] > a) & (srows[:, 0] < b)] if len(srows) else srows
        if len(cand) < 5:
            continue
        # RANSAC leg fit: the candidate cloud is junk-heavy (swinging arms
        # streak too), so sample minimal sets anchored at one endpoint,
        # count agreeing streaks, refit the winners. A gap with a contact
        # can't be one parabola, hence per-endpoint legs.
        rng = np.random.default_rng(int(a))
        end_uv = _project(calib, [out[a], out[b]])
        cam_c = _backproject_at_dist(calib, calib["img_w"] / 2,
                                     calib["img_h"] / 2, 0.0)
        best = None  # (n_inliers, inlier_rows, anchor_row)
        for anchor_f, anchor_uv, anchor_3d in ((a, end_uv[0], out[a]),
                                               (b, end_uv[1], out[b])):
            anchor = np.array([[anchor_f, *anchor_uv, 1, 1.0]])
            depth = float(np.linalg.norm(np.array(anchor_3d) - cam_c))
            for _ in range(40):
                pick = cand[rng.choice(len(cand), 2, replace=False)]
                if len({int(r[0]) for r in pick} | {anchor_f}) < 3:
                    continue
                seg = np.vstack([anchor, pick])
                seg = seg[np.argsort(seg[:, 0])]
                seg[:, 4] = 1.0  # minimal set: all trusted for the probe fit
                # seed from what the pixels say: candidates backprojected at
                # the anchor's depth give position AND spike-grade velocity —
                # the default 3 m/s seed can't climb to 30 m/s legs
                p3 = [_backproject_at_dist(calib, r[1], r[2], depth)
                      for r in pick]
                dt = (pick[1, 0] - pick[0, 0]) / fps
                v_seed = (p3[1] - p3[0]) / dt if dt else np.zeros(3)
                t0 = (seg[0, 0] - pick[0, 0]) / fps
                p_seed = p3[0] + v_seed * t0 if seg[0, 0] != anchor_f \
                    else np.array(anchor_3d)
                x0 = np.concatenate([p_seed, v_seed])
                fit = fit_segment(seg, calib, fps, rms_ok=25.0, x0=x0)
                if fit is None:
                    continue
                p0, v0 = fit[3][:3], fit[3][3:]
                tt = (cand[:, 0] - fit[0][0]) / fps
                path = p0[None] + v0[None] * tt[:, None] + 0.5 * G[None] * tt[:, None] ** 2
                d = np.linalg.norm(_project(calib, path) - cand[:, 1:3], axis=1)
                nin = int((d < 10).sum())
                if nin >= 5 and (best is None or nin > best[0]):
                    best = (nin, cand[d < 10], anchor)
        if best is None:
            continue
        nin, inl, anchor = best
        seg = np.vstack([anchor, np.column_stack([inl[:, :3],
                                                  np.ones(len(inl)),
                                                  np.ones(len(inl))])])
        seg = seg[np.argsort(seg[:, 0])]
        fit = fit_segment(seg, calib, fps, rms_ok=12.0)
        if fit is None:
            continue
        p0, v0 = fit[3][:3], fit[3][3:]
        f_ref = int(fit[0][0])
        lo_f, hi_f = int(seg[0, 0]), int(seg[-1, 0])
        for f in range(max(lo_f, a + 1), min(hi_f, b - 1) + 1):
            t = (f - f_ref) / fps
            pt = p0 + v0 * t + 0.5 * G * t * t
            out[f] = [pt[0], max(pt[1], 0.0), pt[2]]
            filled.add(f)
    return out, n_seg, n_ok, filled


def lift(track, calib, fps, min_len=5, rms_ok=10.0, segmenter="greedy"):
    """Full pipeline: 2D track -> {frame: [x,y,z]}.

    Emits EVERY frame across a fitted segment's span (dense), not just
    detected ones — the ballistic model interpolates interior gaps.
    segmenter: "greedy" (fit-residual growing, legacy) or "velocity"
    (physics-first: cut at velocity breaks, seed across soft touches)."""
    fits = []
    if segmenter == "velocity":
        # consensus anchors are sparse but trusted: let arcs span long
        # detection gaps — the velocity-break test still guards contacts.
        # Track may carry a weight column: segmentation and all acceptance
        # run on w=1 (consensus) rows only; low-weight single-model rows
        # splice into each segment's span as extra pull for the fit.
        if track.shape[1] > 4:
            lo_rows = track[(track[:, 3] > 0) & (track[:, 4] < 0.99)]
            track = track[(track[:, 4] >= 0.99) | (track[:, 3] == 0)]
        else:
            lo_rows = np.empty((0, track.shape[1]))

        def augment(seg):
            if not len(lo_rows) or seg.shape[1] <= 4:
                return seg
            m = (lo_rows[:, 0] > seg[0, 0]) & (lo_rows[:, 0] < seg[-1, 0]) \
                & ~np.isin(lo_rows[:, 0], seg[:, 0])
            if not m.any():
                return seg
            s2 = np.vstack([seg, lo_rows[m]])
            return s2[np.argsort(s2[:, 0])]

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
                seg = augment(seg)
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
        # single-model rescue: dark stretches are often FULL of one-model
        # detections (fast/blurred legs — census showed 20-49 per gap).
        # Let them form arcs under the strict lo-only gate in fit_segment.
        if len(lo_rows):
            spans = [(sa[0, 0], sa[-1, 0]) for sa, _ in pairs]
            free = lo_rows[[not any(a <= f <= b for a, b in spans)
                            for f in lo_rows[:, 0]]]
            for run in runs(free, max_gap=int(fps * 0.5)):
                for fit in grow_segments(run, calib, fps, min_len=8,
                                         rms_ok=8.0):
                    seg = run[np.isin(run[:, 0], fit[0])]
                    pairs.append((seg, fit))
            pairs.sort(key=lambda sf: sf[0][0, 0])
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
        # forward scan, extend-in-place on success: each pair is attempted
        # once, successful merges immediately try to grow further right
        i = 0
        while i < len(pairs) - 1:
            (sa, fa), (sb, fb) = pairs[i], pairs[i + 1]
            if (fa is None and fb is None) or sb[0, 0] - sa[-1, 0] > fps * 0.5:
                i += 1
                continue
            both = np.vstack([sa, sb])
            trial = fit_segment(both, calib, fps, rms_ok, x0=(fa or fb)[3])
            if trial is not None:
                pairs[i:i + 2] = [(both, trial)]   # stay: try growing further
            else:
                i += 1
        pairs = refine_chains([p for p in pairs if p[1] is not None], calib, fps)
        fits = [f for _, f in pairs]
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
                # junk = short AND badly fit. A short arc with pristine rms
                # is a spike leg whose 3D depth merely disagrees with its
                # neighbor — culling those was eating every fast leg.
                if len(fits[short_i][0]) < 10 and fits[short_i][2] > 5.0:
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
    # never-vanish fill (docs/plans/ball-never-vanishes.md): inside a rally
    # the ball is continuous. Gaps between joinable arcs are filled by
    # extrapolating both arcs to their closest approach (the inferred touch);
    # rally-ending arcs continue ballistically to the floor.
    def ext(fit, fr):
        p0, v0 = fit[3][:3], fit[3][3:]
        t = (fr - fit[0][0]) / fps
        return p0 + v0 * t + 0.5 * G * t * t

    filled = set()
    fitted_anchor_frames = {int(f) for fit in fits for f in fit[0]}
    vis_rows = track[track[:, 3] > 0]
    anchor_frames = vis_rows[:, 0].astype(int)
    anchor_xy = {int(r[0]): r[1:3] for r in vis_rows}
    for i, fit in enumerate(fits):
        a_end = int(fit[0][-1])
        nxt = fits[i + 1] if i + 1 < len(fits) else None
        b_start = int(nxt[0][0]) if nxt is not None else None
        joined = False
        # SLOW consensus anchors inside the gap that no arc could explain =
        # a ball that isn't flying (held, serve prep) = rally boundary.
        # FAST unfitted anchors are the opposite: spike-leg evidence (too
        # few frames to fit) — they must not veto the fill.
        stray = 0
        if nxt is not None:
            in_gap = (anchor_frames > a_end) & (anchor_frames < b_start)
            for f in anchor_frames[in_gap]:
                if int(f) in fitted_anchor_frames:
                    continue
                nb = [g for g in (f - 2, f - 1, f + 1, f + 2) if g in anchor_xy]
                if nb:
                    v = max(np.linalg.norm(anchor_xy[int(f)] - anchor_xy[g])
                            / (abs(int(f) - g) / fps) for g in nb)
                    if v > 500:  # px/s: flying, not held
                        continue
                stray += 1
        if nxt is not None and stray < 3 and 1 < b_start - a_end <= fps * 2.5:
            frs = np.arange(a_end + 1, b_start)
            gap_s = len(frs) / fps
            if gap_s <= 1.2:
                # short gap: extrapolate both arcs to the inferred touch
                da = np.array([ext(fit, f) for f in frs])
                db = np.array([ext(nxt, f) for f in frs])
                d = np.linalg.norm(da - db, axis=1)
                k = int(np.argmin(d))
                if d[k] <= 3.0:  # joinable: same rally
                    r = db[k] - da[k]
                    pts = np.array([
                        da[j] + r * ((j + 1) / (k + 1)) / 2 if j <= k
                        else db[j] - r * ((len(frs) - j) / (len(frs) - k)) / 2
                        for j in range(len(frs))])
                    joined = True
            else:
                # long gap: extrapolation diverges — cap it at 0.6s per side
                # and connect the middle with a boundary-value parabola
                # (unique free flight between the two capped endpoints)
                cap = int(0.6 * fps)
                ja, jb = min(cap, len(frs)) - 1, max(len(frs) - cap, 0)
                pa, pb = ext(fit, frs[ja]), ext(nxt, frs[jb])
                T = (frs[jb] - frs[ja]) / fps if jb > ja else 0
                vb = (pb - pa - 0.5 * G * T * T) / T if T else None
                # the connecting flight must be humanly plausible — a fast
                # flat parabola across the gym is a teleport in disguise
                # speed gate scales with brevity: a spike leg is FAST and
                # SHORT (25-35 m/s for <0.6s); fast AND long is a teleport
                v_max = 35.0 if T <= 0.6 else 20.0
                if vb is not None and np.linalg.norm(vb) <= v_max:
                    pts = np.empty((len(frs), 3))
                    for j, f in enumerate(frs):
                        if j <= ja:
                            pts[j] = ext(fit, f)
                        elif j >= jb:
                            pts[j] = ext(nxt, f)
                        else:
                            t = (f - frs[ja]) / fps
                            pts[j] = pa + vb * t + 0.5 * G * t * t
                    joined = True
            if joined:
                # fills own most real net crossings (spike blur = detection
                # gap): if the fill's projection crosses the net band, bend
                # it to cross the plane there, keeping the endpoints fixed
                nx_, ny_ = _net_mid(calib, frame=frs[0])
                fuv = _project(calib, pts)
                finb = (fuv[:, 0] >= nx_[0]) & (fuv[:, 0] <= nx_[-1])
                frel = fuv[:, 1] - np.interp(fuv[:, 0], nx_, ny_)
                for j in range(1, len(pts)):
                    if finb[j] and finb[j - 1] and frel[j - 1] * frel[j] < 0:
                        bump = np.concatenate([
                            np.linspace(0, 1, j + 1),
                            np.linspace(1, 0, len(pts) - j)[1:]])
                        pts[:, 0] -= pts[j, 0] * bump
                        break
                pts = np.clip(pts, [-12, 0, -9], [12, 10, 9])
                for f, p in zip(frs, pts):
                    out[int(f)] = list(p)
                    filled.add(int(f))
        if not joined:
            # rally ends here: continue the last flight to the floor (≤1s),
            # stopping at the gym walls — a ball that exits the volume exits
            # the scene, it doesn't dive through the corner of the frame
            ext_f, ext_p = [], []
            for f in range(a_end + 1, a_end + int(fps) + 1):
                if f in out:
                    break
                p = ext(fit, f)
                if abs(p[0]) > 12 or abs(p[2]) > 9:
                    break
                ext_f.append(f)
                ext_p.append(p)
                if p[1] <= 0:
                    break
            if ext_p:
                pts = np.array(ext_p)
                # point-winning shots cross the net inside this extension:
                # bend x so the crossing sits on the plane (start stays
                # continuous with the arc; landing shifts by the correction)
                nx_, ny_ = _net_mid(calib, frame=ext_f[0])
                euv = _project(calib, pts)
                einb = (euv[:, 0] >= nx_[0]) & (euv[:, 0] <= nx_[-1])
                erel = euv[:, 1] - np.interp(euv[:, 0], nx_, ny_)
                for j in range(1, len(pts)):
                    if einb[j] and einb[j - 1] and erel[j - 1] * erel[j] < 0:
                        ramp = np.concatenate([
                            np.linspace(0, 1, j + 1),
                            np.ones(len(pts) - j - 1)])
                        pts[:, 0] -= pts[j, 0] * ramp
                        break
                for f, p in zip(ext_f, pts):
                    out[f] = [p[0], max(p[1], 0.0), p[2]]
                    filled.add(f)
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
    return out, n_seg, n_ok, filled

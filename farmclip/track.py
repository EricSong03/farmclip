"""Per-frame camera tracking: floor-feature optical flow -> fixed-K solvePnP.

Anchored on one calibrated reference frame. Every tracked feature's world
position lives on the y=0 floor plane (back-projected once through the
anchor pose), so a per-frame PnP solve with constant intrinsics follows
handheld pan/tilt/drift without re-detecting court keypoints.
"""

import cv2
import numpy as np

from .calibrate import calib_matrices, project
from .court import HL, HW

MARGIN = 1.5    # meters of floor kept around the court rectangle
MIN_LIVE = 150  # re-detect features below this count


def pixel_to_floor(calib: dict, uv) -> np.ndarray:
    """Pixels (N,2) -> world (N,3) on the y=0 plane (ray-plane intersection).
    Rays that miss the floor (at/above horizon) come back as NaN rows."""
    K, rvec, tvec = calib_matrices(calib)
    R, _ = cv2.Rodrigues(rvec)
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    rays = R.T @ np.linalg.inv(K) @ np.vstack([uv.T, np.ones(len(uv))])
    origin = (-R.T @ tvec).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -origin[1] / rays[1]
    t = np.where(t > 0, t, np.nan)
    return (origin[:, None] + rays * t).T


def _court_mask(calib: dict, shape) -> np.ndarray:
    rect = [(-HL - MARGIN, 0, -HW - MARGIN), (HL + MARGIN, 0, -HW - MARGIN),
            (HL + MARGIN, 0, HW + MARGIN), (-HL - MARGIN, 0, HW + MARGIN)]
    mask = np.zeros(shape[:2], np.uint8)
    poly = project(calib, rect)
    if np.all(np.isfinite(poly)):
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    return mask


def _detect(gray, mask, exclude=None) -> np.ndarray:
    """goodFeaturesToTrack in mask, avoiding existing points. (N,1,2) f32."""
    m = mask.copy()
    if exclude is not None and len(exclude):
        for p in exclude.reshape(-1, 2):
            cv2.circle(m, (int(p[0]), int(p[1])), 12, 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=400, qualityLevel=0.01,
                                  minDistance=12, mask=m)
    return np.empty((0, 1, 2), np.float32) if pts is None else pts.astype(np.float32)


def _floor_features(calib: dict, gray, exclude=None):
    """Detect features in the court region, back-project to world via calib.
    Returns (pts (N,1,2), world (N,3))."""
    pts = _detect(gray, _court_mask(calib, gray.shape), exclude)
    world = pixel_to_floor(calib, pts.reshape(-1, 2))
    keep = (np.all(np.isfinite(world), axis=1)
            & (np.abs(world[:, 0]) < HL + MARGIN + 1)
            & (np.abs(world[:, 2]) < HW + MARGIN + 1))
    return pts[keep], world[keep]


def _solve(world, pts, calib_prev):
    """PnP with previous pose as guess; RANSAC rescue if the plain solve is
    off. Returns (rvec, tvec, per-point errs) or None."""
    K, rvec0, tvec0 = calib_matrices(calib_prev)
    obj = world.astype(np.float64)
    img = pts.reshape(-1, 2).astype(np.float64)

    def errs(rv, tv):
        proj, _ = cv2.projectPoints(obj, rv, tv, K, None)
        return np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, rvec0.copy(), tvec0.copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        e = errs(rvec, tvec)
        if np.median(e) < 8:
            return rvec, tvec, e
    ok, rvec, tvec, _ = cv2.solvePnPRansac(
        obj, img, K, None, rvec0.copy(), tvec0.copy(), useExtrinsicGuess=True,
        reprojectionError=4.0, flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec, tvec, errs(rvec, tvec)) if ok else None


def _track_pass(frames, ref_gray, calib0, out: dict):
    """Track from the anchor through `frames` (an iterable of (idx, bgr)),
    writing calib dicts into `out`. Direction-agnostic."""
    base = {"img_w": calib0["img_w"], "img_h": calib0["img_h"], "f": calib0["f"]}
    prev_gray = ref_gray
    pts, world = _floor_features(calib0, ref_gray)
    calib = dict(base, rvec=list(calib0["rvec"]), tvec=list(calib0["tvec"]))
    for i, frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        old_gray = prev_gray
        cur, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None,
                                              winSize=(21, 21), maxLevel=3)
        if cur is None:
            st = np.zeros((len(pts), 1), np.uint8)
            cur = pts
            keep = st.ravel() == 1
        else:
            # forward-backward check: bad flow (players, blur, texture
            # aliasing) round-trips far from where it started
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, cur, None,
                                                    winSize=(21, 21), maxLevel=3)
            fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            keep = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        prev_gray = gray
        p_prev, p_cur, w = pts[keep], cur[keep], world[keep]
        if len(p_cur) >= 8:  # homography RANSAC kills flow on moving players
            _, hmask = cv2.findHomography(p_prev, p_cur, cv2.RANSAC, 3.0)
            if hmask is not None:
                keep = hmask.ravel() == 1
                p_cur, w = p_cur[keep], w[keep]
        sol = _solve(w, p_cur, calib) if len(p_cur) >= 6 else None
        if sol is None:
            # ponytail: relock heuristic — after a hard cut assume the shot
            # returns to the anchor camera (tripod-ish footage; zoomed cutaway
            # shots come out wrong either way under constant f). Real per-shot
            # re-anchoring needs court keypoint re-detection.
            cut = float(np.mean(np.abs(gray.astype(np.int16)
                                       - old_gray.astype(np.int16)))) > 25
            if cut:
                calib = dict(base, rvec=list(calib0["rvec"]),
                             tvec=list(calib0["tvec"]))
            print(f"track: frame {i}: solve failed ({len(p_cur)} pts), "
                  f"{'cut -> anchor relock' if cut else 'carry + relock'}")
            out[i] = dict(calib, err=-1.0, n=len(p_cur), carried=True)
            pts, world = _floor_features(calib, gray)
            continue
        rvec, tvec, e = sol
        calib = dict(base, rvec=rvec.ravel().tolist(), tvec=tvec.ravel().tolist())
        inl = e < 8
        rms = float(np.sqrt(np.mean(e[inl] ** 2))) if inl.any() else float(np.max(e))
        out[i] = dict(calib, err=rms, n=int(inl.sum()))
        keep = e < 6  # prune drifted features
        pts, world = p_cur[keep], w[keep]
        if len(pts) < MIN_LIVE:
            fresh, fw = _floor_features(calib, gray, exclude=pts)
            print(f"track: frame {i}: refresh +{len(fresh)} features "
                  f"(live {len(pts)})")
            if len(fresh):
                pts = np.vstack([pts, fresh])
                world = np.vstack([world, fw])


def _frames_forward(video_path, ref_idx, step):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx + 1)
    i = ref_idx + 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (i - ref_idx) % step == 0:
            yield i, frame
        i += 1
    cap.release()


def _frames_backward(video_path, ref_idx, step, chunk=240):
    """ref_idx-1 down to 0, reading forward in chunks (cv2 can't step back)."""
    cap = cv2.VideoCapture(str(video_path))
    hi = ref_idx
    while hi > 0:
        lo = max(0, hi - chunk)
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        buf = []
        for i in range(lo, hi):
            ok, frame = cap.read()
            if not ok:
                break
            buf.append((i, frame))
        for i, frame in reversed(buf):
            if (ref_idx - i) % step == 0:
                yield i, frame
        hi = lo
    cap.release()


def track_camera(video_path, calib0: dict, ref_idx: int = 0, step: int = 1) -> dict:
    """{frame_idx: calib dict} for the whole video, anchored on calib0 at
    ref_idx. f stays constant (same physical camera); only rvec/tvec move."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ok, ref = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {ref_idx} of {video_path}")
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    out = {ref_idx: {"img_w": calib0["img_w"], "img_h": calib0["img_h"],
                     "f": calib0["f"], "rvec": list(calib0["rvec"]),
                     "tvec": list(calib0["tvec"]), "err": 0.0, "n": 0}}
    _track_pass(_frames_forward(video_path, ref_idx, step), ref_gray, calib0, out)
    _track_pass(_frames_backward(video_path, ref_idx, step), ref_gray, calib0, out)
    return out

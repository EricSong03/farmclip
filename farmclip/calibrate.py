"""Camera calibration: named keypoint pixel annotations -> solvePnP pose.

Intrinsics unknown -> grid-search focal length, keep lowest reprojection error.
Annotations for the stationary example clip live in out/annotations.json
(written by Claude reading frames during the loop; regenerate for new videos).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .court import KEYPOINTS, LINES, keypoint_array


def solve(points_2d: dict[str, tuple[float, float]], img_w: int, img_h: int) -> dict:
    """points_2d: keypoint name -> (u, v). Returns calib dict with K, rvec, tvec, err."""
    names = [n for n in points_2d if n in KEYPOINTS]
    if len(names) < 5:
        raise ValueError(f"need >=5 keypoints, got {len(names)}")
    obj = keypoint_array(names)
    img = np.array([points_2d[n] for n in names], dtype=np.float32)

    best = None
    for f_scale in np.arange(0.4, 3.01, 0.05):
        f = f_scale * img_w
        K = np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        if best is None or err < best["err"]:
            best = {"f": f, "K": K, "rvec": rvec, "tvec": tvec, "err": err}
    if best is None:
        raise RuntimeError("solvePnP failed for all focal lengths")

    # refine at the best focal
    ok, rvec, tvec = cv2.solvePnP(
        obj, img, best["K"], None, best["rvec"], best["tvec"],
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    proj, _ = cv2.projectPoints(obj, rvec, tvec, best["K"], None)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    per_point = {
        n: float(np.linalg.norm(proj.reshape(-1, 2)[i] - img[i])) for i, n in enumerate(names)
    }
    return {
        "img_w": img_w, "img_h": img_h,
        "f": best["f"], "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
        "err": err, "per_point_err": per_point,
    }


def lm_polish(calib: dict, points_2d: dict) -> dict:
    """Continuous LM refine over (rvec, tvec, f): solve()'s focal grid is coarse
    (0.05*w steps) and solvePnP alone leaves tens of px on the table.
    Returns polished calib, or the input unchanged if LM doesn't improve."""
    from scipy.optimize import least_squares

    names = [k for k in points_2d if k in KEYPOINTS]
    obj = np.array([KEYPOINTS[k] for k in names], float)
    img = np.array([points_2d[k] for k in names], float)
    w, h = calib["img_w"], calib["img_h"]

    def resid(p):
        rvec, tvec, f = p[:3], p[3:6], p[6]
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        return (proj.reshape(-1, 2) - img).ravel()

    p0 = np.concatenate([calib["rvec"], calib["tvec"], [calib["f"]]])
    # f bounded: unbounded LM escapes to degenerate quasi-orthographic optima
    # (f ~ 1e10) on weakly-conditioned point sets, e.g. single web images
    lo = [-np.inf] * 6 + [0.3 * w]
    hi = [np.inf] * 6 + [3.5 * w]
    p0[6] = np.clip(p0[6], lo[6], hi[6])
    sol = least_squares(resid, p0, method="trf", bounds=(lo, hi), max_nfev=5000)
    new_err = float(np.sqrt(np.mean(np.sum(resid(sol.x).reshape(-1, 2) ** 2, axis=1))))
    if new_err >= calib["err"]:
        return calib
    d = np.linalg.norm(resid(sol.x).reshape(-1, 2), axis=1)
    return dict(calib, rvec=sol.x[:3].tolist(), tvec=sol.x[3:6].tolist(),
                f=float(sol.x[6]), err=new_err,
                per_point_err={k: float(e) for k, e in zip(names, d)})


_NET_GROUP = ("antenna_tip", "net_top", "post_top", "post_base")


def _swap_lr(a: dict, only: str | None = None) -> dict:
    def sw(k):
        if only is not None and not k.startswith(only):
            return k
        if k.endswith("_left"):
            return k[:-5] + "_right"
        if k.endswith("_right"):
            return k[:-6] + "_left"
        return k
    return {sw(k): v for k, v in a.items()}


def _vote_pairs(points_2d: dict, groups=(True, False)) -> dict:
    """Fix individually mirrored L/R pairs: within each group (True=net rig,
    False=floor), majority-vote which image side '_left' lands on and flip
    minority pairs. CAUTION: floor pairs at different depths can legitimately
    have opposite image orderings in corner views — only vote the floor group
    for model detections (solve_auto), never for clicks (solve_web)."""
    pts = dict(points_2d)
    for grp in groups:
        sides = {}
        for k, (u, _) in pts.items():
            if k.endswith("_left") and (k[:-5] in _NET_GROUP) == grp:
                r = pts.get(k[:-5] + "_right")
                if r is not None:
                    sides[k[:-5]] = np.sign(u - r[0])
        maj = np.sign(sum(sides.values())) or 1
        for pre, s in sides.items():
            if s != maj:
                pts[pre + "_left"], pts[pre + "_right"] = \
                    pts[pre + "_right"], pts[pre + "_left"]
    return pts


def solve_auto(points_2d: dict, img_w: int, img_h: int) -> tuple[dict, dict]:
    """solve() for AI-detected keypoints, tolerant of left/right label drift.

    The fine-tuned model inherits per-clip naming drift from its training
    annotations: individual pairs (e.g. net_top) come out mirrored. Two-step
    fix: (1) majority-vote the image side of '_left' within each group (net
    rig vs floor) and flip minority pairs; (2) solve the 4 group-level swap
    variants (as in scripts/calib_solve.py), lowest reprojection error wins.
    Returns (polished calib, the point dict actually used).
    """
    pts = _vote_pairs(points_2d)

    grp_swap = dict(pts)
    for pre in _NET_GROUP:
        grp_swap = _swap_lr(grp_swap, only=pre)
    cands = []
    for a in (pts, _swap_lr(pts), grp_swap, _swap_lr(grp_swap)):
        try:
            c = solve(a, img_w, img_h)
            cands.append((c["err"], c, a))
        except (ValueError, RuntimeError, cv2.error):
            pass
    if not cands:
        raise RuntimeError("solve failed for all left/right variants")
    _, calib, used = min(cands, key=lambda x: x[0])
    return lm_polish(calib, used), used


def _height_at(calib: dict, uv, z0: float) -> float | None:
    """World y of the sideline point (0, y, ±z0) projecting nearest uv.
    Side-agnostic: the click's _left/_right may not match world Z sign."""
    ys = np.arange(0.5, 4.0, 0.002)
    best = (1e9, None)
    for z in (z0, -z0):
        pts = np.stack([np.zeros_like(ys), ys, np.full_like(ys, z)], 1)
        proj = project(calib, pts)
        d = np.hypot(proj[:, 0] - uv[0], proj[:, 1] - uv[1])
        i = int(np.argmin(d))
        if d[i] < best[0]:
            best = (float(d[i]), float(ys[i]))
    return best[1] if best[0] < 80 else None


def solve_web(points_2d: dict, img_w: int, img_h: int) -> tuple[dict, float | None]:
    """Floor-anchored solve for single images (labeling feedback / web data).

    Net height varies by venue (men 2.43, women 2.24, rec anything), so the
    fixed-height net points poison a joint solve. Pose comes from FLOOR clicks
    only (planar IPPE with mirror rejection + bounded LM), then net height is
    MEASURED from the clicked net_top / antenna points. Returns (calib, net_h).
    """
    points_2d = _vote_pairs(points_2d, groups=(True,))  # net rig only, see caution
    floor = {k: v for k, v in points_2d.items()
             if k in KEYPOINTS and KEYPOINTS[k][1] == 0}
    if len(floor) < 5:  # not enough floor: fall back to joint solve
        return lm_polish(solve(points_2d, img_w, img_h), points_2d), None
    from .hypothesis import _solve_points  # deferred: avoids import cycle
    # clicked "_left" may be image-left, i.e. the world mirror — a mirrored
    # floor only fits via an upside-down camera (rejected in _solve_points),
    # so try both assignments and keep the upright one
    cands = []
    for fl in (floor, _swap_lr(floor)):
        names = list(fl)
        c = _solve_points(np.array([KEYPOINTS[n] for n in names], float),
                          np.array([fl[n] for n in names], float),
                          img_w, img_h, f_range=(0.4, 3.0))
        if c is not None:
            cands.append((c["err"], c, fl))
    if cands:
        _, calib, floor = min(cands, key=lambda x: x[0])
        calib["per_point_err"] = {}
        calib = lm_polish(calib, floor)
    else:
        calib = lm_polish(solve(floor, img_w, img_h), floor)
    # joint refine with net-rig clicks: floor anchors the pose, but for
    # low/ground cameras the floor is near edge-on and underdetermines
    # height/tilt — the net verticals are essential constraints there.
    # Net height stays FREE (venues vary: men 2.43, women 2.24, rec anything).
    net = {}
    for n, off in (("net_top_left", 0.0), ("net_top_right", 0.0),
                   ("antenna_tip_left", 0.8), ("antenna_tip_right", 0.8)):
        if n in points_2d:
            uv = np.asarray(points_2d[n], float)
            z0 = KEYPOINTS[n][2]
            # side assignment by nearest projection under the floor-only pose
            da = min(np.hypot(*(project(calib, [(0.0, y, z0)])[0] - uv))
                     for y in (2.0, 2.24, 2.43))
            db = min(np.hypot(*(project(calib, [(0.0, y, -z0)])[0] - uv))
                     for y in (2.0, 2.24, 2.43))
            net[n] = (uv, z0 if da <= db else -z0, off)
    net_h = None
    if net:
        from scipy.optimize import least_squares
        fl_obj = np.array([KEYPOINTS[k] for k in floor], float)
        fl_img = np.array([floor[k] for k in floor], float)
        w_, h_ = img_w, img_h

        def make_resid(nd):
            obj_net = [(z, off) for _, z, off in nd.values()]
            img = np.vstack([fl_img] + [uv for uv, _, _ in nd.values()])

            def resid(p):
                rvec, tvec, f, nh = p[:3], p[3:6], p[6], p[7]
                K = np.array([[f, 0, w_ / 2], [0, f, h_ / 2], [0, 0, 1]])
                obj = np.vstack([fl_obj] + [[(0.0, nh + off, z)] for z, off in obj_net])
                proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
                return (proj.reshape(-1, 2) - img).ravel()
            return resid

        lo = [-np.inf] * 6 + [0.3 * w_, 1.8]
        hi = [np.inf] * 6 + [3.5 * w_, 2.7]
        # multi-start: degenerate floor-only inits (ground-level cameras see
        # the floor edge-on) strand a single TRF run in the wrong basin; the
        # net side assignment can also be wrong under a bad init.
        variants = [net, {k: (uv, -z, off) for k, (uv, z, off) in net.items()}]
        best = None
        for nd in variants:
            resid = make_resid(nd)
            for f0 in {np.clip(calib["f"], 0.3 * w_, 3.5 * w_), 0.8 * w_, 1.5 * w_, 2.5 * w_}:
                p0 = np.concatenate([calib["rvec"], calib["tvec"], [f0], [2.43]])
                try:
                    sol = least_squares(resid, p0, method="trf",
                                        bounds=(lo, hi), max_nfev=3000)
                except Exception:
                    continue
                cost = float(np.sqrt(np.mean(sol.fun ** 2)))
                if best is None or cost < best[0]:
                    best = (cost, sol.x.copy(), resid)
        if best is not None:
            _, x, resid = best
            net_h = float(x[7])
            fl_res = resid(x).reshape(-1, 2)[:len(fl_obj)]
            calib = dict(calib, rvec=x[:3].tolist(), tvec=x[3:6].tolist(),
                         f=float(x[6]),
                         err=float(np.sqrt(np.mean(np.sum(fl_res ** 2, axis=1)))),
                         net_h_est=round(net_h, 3))
    return calib, net_h


def calib_matrices(calib: dict):
    K = np.array(
        [[calib["f"], 0, calib["img_w"] / 2], [0, calib["f"], calib["img_h"] / 2], [0, 0, 1]]
    )
    return K, np.array(calib["rvec"]).reshape(3, 1), np.array(calib["tvec"]).reshape(3, 1)


def project(calib: dict, pts3d) -> np.ndarray:
    """(N,3) world meters -> (N,2) pixels."""
    K, rvec, tvec = calib_matrices(calib)
    proj, _ = cv2.projectPoints(np.asarray(pts3d, dtype=np.float64), rvec, tvec, K, None)
    return proj.reshape(-1, 2)


def consistent_names(ann: dict, calib: dict, max_px: float = 200.0) -> dict:
    """Reassign clicked-point names to the nearest calib-projected keypoint.

    Clicks are precise but _left/_right naming conventions drifted between
    clips; the solved calib is convention-correct, so names come from
    projection, positions stay human. Greedy 1-1 by distance; clicks with no
    projection within max_px are dropped (printed).
    """
    names = list(KEYPOINTS)
    proj = project(calib, np.array([KEYPOINTS[n] for n in names], np.float32))
    clicks = list(ann.items())
    pairs = sorted(
        (float(np.hypot(uv[0] - p[0], uv[1] - p[1])), ci, pi)
        for ci, (_, uv) in enumerate(clicks) for pi, p in enumerate(proj))
    used_c, used_p, out = set(), set(), {}
    for d, ci, pi in pairs:
        if ci in used_c or pi in used_p or d > max_px:
            continue
        used_c.add(ci), used_p.add(pi)
        cn, uv = clicks[ci]
        if names[pi] != cn:
            print(f"  rename {cn} -> {names[pi]} ({d:.0f}px)")
        out[names[pi]] = uv
    for ci, (cn, _) in enumerate(clicks):
        if ci not in used_c:
            print(f"  drop {cn} (no projection within {max_px:.0f}px)")
    return out


def draw_overlay(frame, calib: dict, annotations: dict | None = None,
                 net_h: float | None = None):
    """Court lines (cyan), keypoints (red +), annotations (green x) on a copy of frame.
    net_h: measured net height — shifts everything above the floor by
    (net_h - regulation 2.43) so the net band/antennas draw where they really are."""
    out = frame.copy()
    dy = 0.0 if net_h is None else net_h - 2.43
    shift = lambda p: (p[0], p[1] + dy, p[2]) if p[1] > 0 else p
    for a, b in [(shift(a), shift(b)) for a, b in LINES]:
        seg3d = np.linspace(a, b, 20)  # sample so long lines curve correctly under perspective? (pinhole: straight, but keeps clipping simple)
        pts = project(calib, seg3d)
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            if np.all(np.isfinite(p)) and np.all(np.isfinite(q)):
                cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), (255, 255, 0), 2)
    for name, (x, y, z) in KEYPOINTS.items():
        (u, v), = project(calib, [(x, y + dy if y > 0 else y, z)])
        if 0 <= u < out.shape[1] and 0 <= v < out.shape[0]:
            cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(out, name, (int(u) + 4, int(v) - 4), 0, 0.35, (0, 0, 255), 1)
    if annotations:
        for name, (u, v) in annotations.items():
            cv2.drawMarker(out, (int(u), int(v)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
    return out


def run(annotations_path: str | Path, frame_path: str | Path, out_dir: str | Path = "out") -> dict:
    """Solve from an annotations JSON + frame image; write calib.json + overlay."""
    out_dir = Path(out_dir)
    ann = json.loads(Path(annotations_path).read_text())
    frame = cv2.imread(str(frame_path))
    h, w = frame.shape[:2]
    calib = solve(ann, w, h)
    (out_dir / "calib.json").write_text(json.dumps(calib, indent=1))
    dbg = out_dir / "debug"
    dbg.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dbg / "calib_overlay.jpg"), draw_overlay(frame, calib, ann))
    return calib

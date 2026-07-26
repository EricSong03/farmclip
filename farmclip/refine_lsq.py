"""Nonlinear calibration polish: scipy least_squares over
(f, rvec, tvec, net_h, post_hw), minimizing point-to-line distances for floor
lines + net band and pixel error for post bases.

Court floor dims are regulation-certain; net rig geometry (band height as
strung, post distance from sidelines) varies by gym -> solved, bounded.
"""

import numpy as np
import cv2
from scipy.optimize import least_squares


def _project(K, rvec, tvec, pts3d):
    proj, _ = cv2.projectPoints(np.asarray(pts3d, float), rvec, tvec, K, None)
    return proj.reshape(-1, 2)


def _line_res(seg, pts2d, h):
    x1, y1, x2, y2 = seg
    n = np.array([-(y2 - y1), x2 - x1], float)
    n /= np.linalg.norm(n)
    lo, hi = min(x1, x2) - 40, max(x1, x2) + 40
    out = []
    for p in pts2d:
        if not np.all(np.isfinite(p)):
            out.append(50.0)
        elif lo <= p[0] <= hi and -100 <= p[1] <= h + 100:
            out.append(float((p - [x1, y1]) @ n))
        else:
            out.append(0.0)
    return out


def _residuals(params, w, h, floor_pairs, band_seg, post_px):
    f, rx, ry, rz, tx, ty, tz, net_h, post_hw, sx, sz = params
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
    rvec = np.array([rx, ry, rz])
    tvec = np.array([tx, ty, tz])
    scale = np.array([sx, 1.0, sz])  # non-regulation courts: scale floor x/z
    res = []
    for seg, (a, b) in floor_pairs:
        pts = np.linspace(a, b, 12) * scale
        res += _line_res(seg, _project(K, rvec, tvec, pts), h)
    if band_seg is not None:
        band3d = np.linspace((0, net_h, -post_hw), (0, net_h, +post_hw), 12)
        res += _line_res(band_seg, _project(K, rvec, tvec, band3d), h)
    for pt2d, z_sign in post_px:
        p = _project(K, rvec, tvec, [(0, 0, z_sign * post_hw)])[0]
        d = (p - np.asarray(pt2d, float)) * 5.0
        res += ([float(d[0]), float(d[1])] if np.all(np.isfinite(d)) else [250.0, 250.0])
    return np.array(res)


def polish(calib, floor_pairs, band_seg=None, post_px=()):
    """post_px: [((u, v), z_sign), ...] measured post bases; z_sign +1 = image-left post."""
    w, h = calib["img_w"], calib["img_h"]
    x0 = np.array([calib["f"], *calib["rvec"], *calib["tvec"], 2.43, 4.75, 1.0, 1.0], float)
    out = least_squares(
        _residuals, x0, args=(w, h, floor_pairs, band_seg, post_px),
        loss="soft_l1", f_scale=4.0, max_nfev=800,
        # dims near-regulation: +-7% absorbs noise without letting a wrong line
        # assignment warp the court to fit (courts ARE regulation; fix the
        # assignment, not the court)
        # court dims LOCKED to regulation (18x9 is certain; freeing them lets a
        # wrong line assignment warp the court to fit at near-zero error).
        # net_h floor 2.0: rec nets are often strung below standard height.
        bounds=([0.4 * w, -np.pi, -np.pi, -np.pi, -60, -60, -60, 2.00, 4.55, 0.999, 0.999],
                [2.0 * w, np.pi, np.pi, np.pi, 60, 60, 60, 2.55, 5.60, 1.001, 1.001]),
    )
    f, rx, ry, rz, tx, ty, tz, net_h, post_hw, sx, sz = out.x
    r = _residuals(out.x, w, h, floor_pairs, band_seg, post_px)
    from .court import COURT_L, COURT_W
    return {
        "img_w": w, "img_h": h, "f": float(f),
        "rvec": [rx, ry, rz], "tvec": [tx, ty, tz],
        "net_h": float(net_h), "post_hw": float(post_hw),
        "court_l": float(COURT_L * sx), "court_w": float(COURT_W * sz),
        "err": float(np.sqrt(np.mean(r ** 2))), "per_point_err": {},
    }

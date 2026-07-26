"""Line-based calibration: ICP-style alternation between point association
and solvePnP, using detected 2D segments matched to 3D court-model lines.

Each iteration: sample the 2D segments, pair each sample with the closest
point on its 3D model line under the current pose, re-solve PnP. Converges in
a handful of rounds given a roughly-correct init.
"""

import numpy as np
import cv2

from .calibrate import calib_matrices, project


def _sample_seg(seg, n=15):
    x1, y1, x2, y2 = seg
    t = np.linspace(0, 1, n)
    return np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1)


def _closest_on_line3d(calib, a, b, pt2d, n=200):
    """3D point on segment a-b whose projection is nearest to pt2d."""
    t = np.linspace(0, 1, n)
    pts3d = np.array(a)[None] + t[:, None] * (np.array(b) - np.array(a))[None]
    proj = project(calib, pts3d)
    d = np.linalg.norm(proj - pt2d, axis=1)
    return pts3d[int(np.argmin(d))], float(d.min())


def refine(calib, line_pairs, point_pairs=None, iters=8, f_range=(0.4, 3.0)):
    """line_pairs: [(seg2d, (a3d, b3d)), ...]; point_pairs: {name-free: (pt2d, pt3d)}.

    Returns refined calib dict (same shape as calibrate.solve output).
    """
    img_w, img_h = calib["img_w"], calib["img_h"]
    point_pairs = point_pairs or []
    for _ in range(iters):
        obj, img = [], []
        for pt2d, pt3d in point_pairs:
            obj.append(pt3d)
            img.append(pt2d)
        for seg, (a, b) in line_pairs:
            for s in _sample_seg(seg):
                p3d, _ = _closest_on_line3d(calib, a, b, s)
                obj.append(p3d)
                img.append(s)
        obj = np.array(obj, dtype=np.float64)
        img = np.array(img, dtype=np.float64)

        best = None
        for f_scale in np.arange(f_range[0], f_range[1] + 0.01, 0.05):
            f = f_scale * img_w
            K = np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1]])
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
            if best is None or err < best[0]:
                best = (err, f, rvec, tvec)
        err, f, rvec, tvec = best
        calib = {
            "img_w": img_w, "img_h": img_h, "f": f,
            "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
            "err": err, "per_point_err": {},
        }
    return calib

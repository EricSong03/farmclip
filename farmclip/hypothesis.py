"""RANSAC-style calibration: enumerate segment->model-line assignments,
solve each, keep the hypothesis whose projection explains the most segments
while passing physical sanity gates.

Anchors (high confidence, given by caller): near attack line + net band.
Hypothesized: left/right sidelines (diagonal segs) x one far floor line.
"""

import itertools

import cv2
import numpy as np

from .court import HL, HW, ATTACK, NET_H, LINES
from .lines import intersect
from .calibrate import project


FAR_LINES = {
    "attack_far": ((-ATTACK, 0, -HW), (-ATTACK, 0, +HW)),
    "end_far": ((-HL, 0, -HW), (-HL, 0, +HW)),
    "center": ((0, 0, -HW), (0, 0, +HW)),
}


def _solve_points(obj, img, w, h, f_range=(0.5, 1.6)):
    """Planar PnP with focal search. Uses IPPE (both mirror solutions) and
    keeps the physically sane one (camera above the floor)."""
    # IPPE needs the planar object in z=0; our floor is y=0 (Y-up world).
    # Rotate world->planar by -90deg about X: (x,y,z) -> (x,z,-y), then map
    # the solved pose back via R_world = R_planar @ M.
    M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    obj_p = (M @ np.asarray(obj, float).T).T
    best = None
    for fs in np.arange(*f_range, 0.05):
        f = fs * w
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj_p, np.asarray(img, float), K, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        for rvec_p, tvec in zip(rvecs, tvecs):
            R_p, _ = cv2.Rodrigues(rvec_p)
            R = R_p @ M
            rvec, _ = cv2.Rodrigues(R)
            C = (-R.T @ tvec).ravel()
            if C[1] <= 0:  # camera below floor = mirror ghost
                continue
            proj, _ = cv2.projectPoints(np.asarray(obj, float), rvec, tvec, K, None)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
            if best is None or err < best["err"]:
                best = {"img_w": w, "img_h": h, "f": f, "rvec": rvec.ravel().tolist(),
                        "tvec": tvec.ravel().tolist(), "err": err, "per_point_err": {}}
    return best


def _cam_center(calib):
    R, _ = cv2.Rodrigues(np.array(calib["rvec"]))
    return (-R.T @ np.array(calib["tvec"]).reshape(3, 1)).ravel()


def _sane(calib):
    C = _cam_center(calib)
    if not (1.5 <= C[1] <= 12):        # elevated but not on the ceiling
        return False
    if not (HL - 3 <= abs(C[0]) <= 40):  # behind an end line-ish (either side; court is symmetric)
        return False
    # far floor lines must project BELOW the net top band on screen
    far = project(calib, [(-HL, 0, 0)])[0]
    net = project(calib, [(0, NET_H, 0)])[0]
    if not (np.all(np.isfinite(far)) and np.all(np.isfinite(net))):
        return False
    return far[1] > net[1]


def _seg_support(calib, segs, tol=10):
    """How many detected segments lie near ANY projected model line."""
    model_pts = []
    for a, b in LINES:
        model_pts.append(project(calib, np.linspace(a, b, 40)))
    model_pts = np.vstack(model_pts)
    n_in, support = 0, 0.0
    for x1, y1, x2, y2 in segs:
        s = np.stack([np.linspace(x1, x2, 8), np.linspace(y1, y2, 8)], 1)
        d = np.linalg.norm(s[:, None] - model_pts[None], axis=2).min(1)
        if np.median(d) < tol:
            n_in += 1
            support += np.hypot(x2 - x1, y2 - y1)
    return n_in, support


def search_headon(segs, net_band, w, h, verbose=False):
    """Head-on (end-line, centered) camera: sidelines are invisible slivers, but
    both net posts are. Anchor on the net plane: post verticals x net band ->
    4 points in the x=0 plane -> planar PnP; a low horizontal floor line
    (tried as near end line and near attack line) picks depth via refine_lsq.
    Returns (calib, info) or (None, None)."""
    from .refine_lsq import polish

    def slope(s):
        return abs(s[3] - s[1]) / max(abs(s[2] - s[0]), 1)

    bx_lo, bx_hi = min(net_band[0], net_band[2]), max(net_band[0], net_band[2])
    by = (net_band[1] + net_band[3]) / 2
    verts = [s for s in segs if slope(s) > 3 and abs(s[3] - s[1]) > 40
             and bx_lo - 60 <= (s[0] + s[2]) / 2 <= bx_hi + 60
             and max(s[1], s[3]) > by]  # extends below the band = a post, not rigging
    if len(verts) < 2:
        return None, None
    verts.sort(key=lambda s: (s[0] + s[2]) / 2)
    postL, postR = verts[0], verts[-1]
    if (postR[0] + postR[2]) / 2 - (postL[0] + postL[2]) / 2 < w * 0.3:
        return None, None
    HW_POST = 4.75  # nominal; polish frees it

    def post_pts(seg, z_sign):
        top = intersect(seg, net_band)
        base = max([(seg[0], seg[1]), (seg[2], seg[3])], key=lambda p: p[1])
        return [(top, (0, NET_H, z_sign * HW_POST)), (base, (0, 0, z_sign * HW_POST))]

    pairs = post_pts(postL, +1) + post_pts(postR, -1)  # image-left = +Z
    if any(p[0] is None for p in pairs):
        return None, None
    obj = np.array([p[1] for p in pairs], float)
    img = np.array([p[0] for p in pairs], float)
    # planar PnP in the x=0 plane: rotate about Y so the net plane becomes z=0
    M = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    calib0 = None
    from .calibrate import calib_matrices  # noqa: F401  (parity with _solve_points internals)
    best0 = None
    for fs in np.arange(0.5, 1.6, 0.05):
        f = fs * w
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                (M @ obj.T).T, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        for rvec_p, tvec in zip(rvecs, tvecs):
            R = cv2.Rodrigues(rvec_p)[0] @ M
            C = (-R.T @ tvec).ravel()
            if C[1] <= 0.3 or abs(C[0]) < 2:
                continue
            rvec = cv2.Rodrigues(R)[0]
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
            if best0 is None or err < best0["err"]:
                best0 = {"img_w": w, "img_h": h, "f": f, "rvec": rvec.ravel().tolist(),
                         "tvec": tvec.ravel().tolist(), "err": err, "per_point_err": {}}
    if best0 is None:
        return None, None

    horiz = [s for s in segs if slope(s) < 0.15 and (s[1] + s[3]) / 2 > by + 60
             and abs(s[2] - s[0]) > w * 0.4]
    best = None
    post_px = [(pairs[1][0], +1), (pairs[3][0], -1)]
    for segF in horiz:
        for far_name in ("end_near", "attack_near"):
            fx = HL if far_name == "end_near" else ATTACK
            floor_pairs = [(segF, ((fx, 0, -HW), (fx, 0, +HW)))]
            calib = polish(dict(best0), floor_pairs, band_seg=net_band, post_px=post_px)
            C = _cam_center(calib)
            if not (0.5 <= C[1] <= 6 and abs(C[0]) > HL / 2 - 2):
                continue
            n_in, support = _seg_support(calib, segs)
            key = (n_in, support, -calib["err"])
            if best is None or key > best[0]:
                best = (key, calib, {"family": "headon", "floor": far_name,
                                     "n_inliers": n_in, "err": calib["err"]})
                if verbose:
                    print("headon best", best[2])
    if best is None:
        return None, None
    return best[1], best[2]


def search(segs, attack_near, net_band, w, h, verbose=False):
    """Try all (segL, segR, segF->far model line) hypotheses. Returns (calib, info)."""
    def slope(s):
        return abs(s[3] - s[1]) / max(abs(s[2] - s[0]), 1)

    def ymid(s):
        return (s[1] + s[3]) / 2

    diag = [s for s in segs if 0.2 < slope(s) < 5 and ymid(s) > 250]
    horiz = [s for s in segs if slope(s) < 0.25 and 280 < ymid(s) < 430]
    best, best_key, best_info = None, None, None
    for segL, segR in itertools.permutations(diag, 2):
        if ymid(segL) < 100 or (segL[0] + segL[2]) >= (segR[0] + segR[2]):
            continue  # L must be left of R
        anl = intersect(attack_near, segL)
        anr = intersect(attack_near, segR)
        if anl is None or anr is None:
            continue
        for far_name, far3d in FAR_LINES.items():
            for segF in horiz:
                if np.hypot(segF[2] - segF[0], segF[3] - segF[1]) < 100:
                    continue  # noise marks can't anchor perspective
                fl = intersect(segF, segL)
                fr = intersect(segF, segR)
                if fl is None or fr is None:
                    continue
                fx = far3d[0][0]
                # image-left = world +Z for a +X-side camera (chirality!)
                obj = np.array([
                    (ATTACK, 0, +HW), (ATTACK, 0, -HW),
                    (fx, 0, +HW), (fx, 0, -HW),
                ], float)
                img = np.array([anl, anr, fl, fr], float)
                if np.any(img[:, 0] < -200) or np.any(img[:, 0] > w + 200):
                    continue
                calib = _solve_points(obj, img, w, h)
                if calib is None or calib["err"] > 30 or not _sane(calib):
                    continue
                # hard gate: projected net top band must lie on the detected band
                band3d = np.linspace((0, NET_H, +HW), (0, NET_H, -HW), 10)
                bp = project(calib, band3d)
                bx1, by1, bx2, by2 = net_band
                nvec = np.array([-(by2 - by1), bx2 - bx1], float)
                nvec /= np.linalg.norm(nvec)
                d = np.abs((bp - [bx1, by1]) @ nvec)
                if np.median(d) > 40:  # loose gate; line-ICP tightens after
                    continue
                n_in, support = _seg_support(calib, segs)
                key = (n_in, support, -calib["err"])
                if best_key is None or key > best_key:
                    best_key = key
                    best = calib
                    best_info = {"far": far_name, "n_inliers": n_in,
                                 "support": support, "err": calib["err"],
                                 "segL": list(segL), "segR": list(segR), "segF": list(segF)}
                    if verbose:
                        print("new best", best_info)
    return best, best_info

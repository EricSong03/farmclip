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
    sol = least_squares(resid, p0, method="lm", max_nfev=5000)
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


def solve_auto(points_2d: dict, img_w: int, img_h: int) -> tuple[dict, dict]:
    """solve() for AI-detected keypoints, tolerant of left/right label drift.

    The fine-tuned model inherits per-clip naming drift from its training
    annotations: individual pairs (e.g. net_top) come out mirrored. Two-step
    fix: (1) majority-vote the image side of '_left' within each group (net
    rig vs floor) and flip minority pairs; (2) solve the 4 group-level swap
    variants (as in scripts/calib_solve.py), lowest reprojection error wins.
    Returns (polished calib, the point dict actually used).
    """
    pts = dict(points_2d)
    for grp in (True, False):  # True = net-rig pairs, False = floor pairs
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


def draw_overlay(frame, calib: dict, annotations: dict | None = None):
    """Court lines (cyan), keypoints (red +), annotations (green x) on a copy of frame."""
    out = frame.copy()
    for a, b in LINES:
        seg3d = np.linspace(a, b, 20)  # sample so long lines curve correctly under perspective? (pinhole: straight, but keeps clipping simple)
        pts = project(calib, seg3d)
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            if np.all(np.isfinite(p)) and np.all(np.isfinite(q)):
                cv2.line(out, tuple(p.astype(int)), tuple(q.astype(int)), (255, 255, 0), 2)
    for name, (x, y, z) in KEYPOINTS.items():
        (u, v), = project(calib, [(x, y, z)])
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

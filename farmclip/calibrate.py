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

"""Players v2: 3D skeletons on CPU. YOLO11-pose 2D keypoints -> 3D lift.

Lift v1 is geometric ("billboard-plane"): each player's joints are placed on
the vertical plane through their ground position, facing the camera. Depth
within a body is small vs camera distance, so this looks right from most
viewing angles.
# ponytail: billboard lift; upgrade path = VideoPose3D-class temporal lifter
# (pretrained, CPU-fast) when edge-on views look too flat.
"""

import numpy as np
import cv2

# COCO-17 keypoint order (YOLO pose output)
COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"]


def _cam(calib):
    K = np.array([[calib["f"], 0, calib["img_w"] / 2],
                  [0, calib["f"], calib["img_h"] / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array(calib["rvec"], float))
    t = np.array(calib["tvec"], float).reshape(3, 1)
    C = (-R.T @ t).ravel()
    Kinv_RT = R.T @ np.linalg.inv(K)
    return C, Kinv_RT


def lift_billboard(kpts_2d, conf, ground_xz, calib, min_conf=0.35):
    """kpts_2d: (17,2) pixels; ground_xz: player's floor position.

    Intersect each joint's camera ray with the vertical plane through the
    ground point whose normal is the horizontal camera->player direction.
    Returns (17,3) world coords with NaN rows for low-confidence joints.
    """
    C, Kinv_RT = _cam(calib)
    px, pz = ground_xz
    n = np.array([px - C[0], 0.0, pz - C[2]])
    nn = np.linalg.norm(n)
    if nn < 1e-6:
        return None
    n /= nn
    p0 = np.array([px, 0.0, pz])
    out = np.full((17, 3), np.nan)
    for i, (u, v) in enumerate(kpts_2d):
        if conf[i] < min_conf:
            continue
        d = Kinv_RT @ np.array([u, v, 1.0])
        denom = d @ n
        if abs(denom) < 1e-9:
            continue
        s = ((p0 - C) @ n) / denom
        if s <= 0:
            continue
        p = C + s * d
        if -0.5 < p[1] < 3.6:  # joints live between floor and reach height
            out[i] = p
    return out


def joints_for_scene(result_kpts, tracked, players, calib):
    """Match YOLO pose results to on-court players; add 'joints' per player.

    result_kpts: list of (kpts (17,2), conf (17,)) per detection, same order
    as the boxes passed to the tracker this frame.
    tracked: [(tid, box), ...] aligned with result_kpts by index? No —
    tracker reorders; match by box identity instead.
    players: scene player dicts (id like 'a12') to enrich in place.
    """
    by_tid = {}
    for (tid, box) in tracked:
        for kpts, conf, kbox in result_kpts:
            if all(abs(a - b) < 1.0 for a, b in zip(box, kbox)):
                by_tid[tid] = (kpts, conf)
                break
    for p in players:
        tid = int(p["id"][1:])
        if tid not in by_tid:
            continue
        kpts, conf = by_tid[tid]
        j = lift_billboard(kpts, conf, p["pos"], calib)
        if j is None:
            continue
        p["joints"] = [[round(float(v), 3) for v in row] if np.isfinite(row).all() else None
                       for row in j]

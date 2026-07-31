"""Court-line segmentation calibration (Phase 3, docs/plans/line-seg-calib.md).

A 9-class UNet (scripts/train_lineseg.py) segments the named court lines; the
pose is solved by minimizing point-to-projected-LINE distance rather than
point-to-point. Lines beat corner keypoints on unseen venues: they stay in
frame, survive occlusion, and every pixel of a line is a constraint.

Residuals use the interpretation-plane form (l = K^-T (a_cam x b_cam)) instead
of projecting endpoints — correct when a line endpoint is behind the camera,
which happens constantly for sidelines.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .calibrate import calib_matrices, project
from .court import ATTACK, HL, HW

CLASSES = ["bg", "sideline_left", "sideline_right", "endline_far", "endline_near",
           "attack_far", "attack_near", "center_line", "net_band"]
NET_EXT = 0.9  # net band overhangs the sidelines to the posts-ish
MAX_PX = 250   # per class, so a long sideline can't outvote a short attack line

_SESSIONS = {}


def class_lines(net_h: float) -> dict[int, tuple]:
    """class id -> 3D segment. Left = +Z = IMAGE-left (court.py convention)."""
    return {
        # sideline_left is +Z, matching court.KEYPOINTS: left means IMAGE-left.
        1: ((-HL, 0, +HW), (+HL, 0, +HW)),
        2: ((-HL, 0, -HW), (+HL, 0, -HW)),
        3: ((-HL, 0, -HW), (-HL, 0, +HW)),
        4: ((+HL, 0, -HW), (+HL, 0, +HW)),
        5: ((-ATTACK, 0, -HW), (-ATTACK, 0, +HW)),
        6: ((+ATTACK, 0, -HW), (+ATTACK, 0, +HW)),
        7: ((0, 0, -HW), (0, 0, +HW)),
        8: ((0, net_h, -HW - NET_EXT), (0, net_h, +HW + NET_EXT)),
    }


def _session(model_path: str):
    if model_path not in _SESSIONS:
        import onnxruntime as ort
        _SESSIONS[model_path] = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
    return _SESSIONS[model_path]


def detect_lines(frames, model_path="finetune_out/lineseg/lineseg.onnx",
                 min_px: int = 60, rng=None) -> dict[int, np.ndarray]:
    """BGR frame(s) -> {class id: (N,2) pixel coords} in ORIGINAL image pixels.

    Averages softmax over the frames before argmax: players and refs occlude
    different lines in different frames, so the mean mask is cleaner than any
    single one (camera is stationary — that's the whole premise of this repo).
    """
    frames = [frames] if isinstance(frames, np.ndarray) else list(frames)
    sess = _session(str(model_path))
    inp = sess.get_inputs()[0]
    _, _, mh, mw = inp.shape
    h, w = frames[0].shape[:2]
    prob = None
    for f in frames:
        img = cv2.resize(f, (mw, mh))  # training used plain resize, no letterbox
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        logits = sess.run(None, {inp.name: np.ascontiguousarray(x)})[0][0]
        e = np.exp(logits - logits.max(0, keepdims=True))
        prob = e / e.sum(0) if prob is None else prob + e / e.sum(0)
    lab = prob.argmax(0)

    rng = rng or np.random.default_rng(0)
    sx, sy = w / mw, h / mh
    out = {}
    for cid in range(1, len(CLASSES)):
        ys, xs = np.nonzero(lab == cid)
        if len(xs) < min_px:
            continue
        if len(xs) > MAX_PX:
            k = rng.choice(len(xs), MAX_PX, replace=False)
            ys, xs = ys[k], xs[k]
        out[cid] = np.stack([(xs + 0.5) * sx, (ys + 0.5) * sy], 1)
    return out


def line_dists(calib: dict, pix: dict[int, np.ndarray], net_h: float) -> dict[int, np.ndarray]:
    """Per class, perpendicular pixel distance from each sample to its projected line."""
    K, rvec, tvec = calib_matrices(calib)
    R = cv2.Rodrigues(rvec)[0]
    t = tvec.ravel()
    KinvT = np.linalg.inv(K).T
    out = {}
    for cid, pts in pix.items():
        a, b = class_lines(net_h)[cid]
        n = np.cross(R @ a + t, R @ b + t)  # interpretation plane normal (camera frame)
        l = KinvT @ n                        # image line: l . [u,v,1] = 0
        d = np.hypot(l[0], l[1])
        if d < 1e-9:                         # camera on the line — degenerate
            out[cid] = np.full(len(pts), 1e3)
            continue
        out[cid] = (pts @ l[:2] + l[2]) / d
    return out


def solve_lines(pix: dict[int, np.ndarray], init: dict, net_h: float = 2.43,
                free_net_h: bool = True) -> tuple[dict, float]:
    """Refine (rvec, tvec, f, net_h) against segmented line pixels.

    init: any rough calib (AI keypoints, hough, or a previous solve) — the line
    solve is a refiner, not a global search; it has no notion of which end of
    the court is which. Returns (calib, median residual px).
    """
    from scipy.optimize import least_squares

    w, h = init["img_w"], init["img_h"]
    order = sorted(pix)

    def resid(p):
        c = dict(init, rvec=p[:3].tolist(), tvec=p[3:6].tolist(), f=float(p[6]))
        d = line_dists(c, pix, float(p[7]))
        # equal weight per class: pixel counts differ 10x between lines
        return np.concatenate([d[cid] / np.sqrt(len(d[cid])) for cid in order])

    p0 = np.concatenate([init["rvec"], init["tvec"],
                         [np.clip(init["f"], 0.3 * w, 3.5 * w)], [net_h]])
    nh_lo, nh_hi = (1.8, 2.7) if free_net_h else (net_h - 1e-6, net_h + 1e-6)
    lo = [-np.inf] * 6 + [0.3 * w, nh_lo]
    hi = [np.inf] * 6 + [3.5 * w, nh_hi]
    p0[7] = np.clip(p0[7], nh_lo, nh_hi)
    # soft_l1: segmentation always leaks a few pixels onto floor markings from
    # other sports — plain least squares lets those drag the pose
    sol = least_squares(resid, p0, method="trf", bounds=(lo, hi),
                        loss="soft_l1", f_scale=5.0, max_nfev=3000)
    calib = dict(init, rvec=sol.x[:3].tolist(), tvec=sol.x[3:6].tolist(),
                 f=float(sol.x[6]), net_h_est=round(float(sol.x[7]), 3),
                 method="lineseg")
    nh = float(sol.x[7])
    d = line_dists(calib, pix, nh)
    med = float(np.median(np.abs(np.concatenate([d[c] for c in order]))))
    calib["err"] = med
    calib["per_line_err"] = {CLASSES[c]: round(float(np.median(np.abs(d[c]))), 2)
                             for c in order}
    # Net height is unobservable when the camera sits in the net plane (x=0):
    # every height then projects to the SAME image line. Sideline cameras sit
    # near that plane, so check sensitivity instead of trusting the optimum.
    if 8 not in pix:
        calib.pop("net_h_est", None)
    elif min(abs(nh - nh_lo), abs(nh - nh_hi)) < 1e-3:
        # pinned to a bound = the optimizer ran out of room, not a measurement
        calib.pop("net_h_est", None)
    else:
        base = np.median(np.abs(d[8]))
        swing = max(np.median(np.abs(line_dists(calib, {8: pix[8]}, nh + s)[8])) - base
                    for s in (-0.15, 0.15))
        if swing < 1.0:
            calib.pop("net_h_est", None)  # pose is fine, the height is not measured
    return calib, med


def draw_mask(frame, pix: dict[int, np.ndarray]):
    """Segmented line pixels colour-coded by class, on a copy of frame."""
    out = frame.copy()
    for cid, pts in pix.items():
        col = tuple(int(v) for v in cv2.applyColorMap(
            np.uint8([[cid * 28]]), cv2.COLORMAP_HSV)[0, 0])
        for u, v in pts.astype(int):
            cv2.circle(out, (u, v), 2, col, -1)
    return out


def _sample(truth, net_h, n=60):
    pix = {}
    for cid, (a, b) in class_lines(net_h).items():
        uv = project(truth, np.linspace(a, b, n))
        pix[cid] = uv[np.isfinite(uv).all(1)]
    return pix


def _self_check():
    """Residuals must vanish for pixels sampled off a known calib's own lines,
    and a perturbed init must converge back onto it."""
    net_h = 2.31
    # off-axis camera: net height IS observable here
    truth = {"img_w": 1280, "img_h": 720, "f": 1100.0,
             "rvec": [1.75, 0.55, 0.05], "tvec": [1.5, 2.0, 16.0], "err": 0.0}
    pix = _sample(truth, net_h)
    worst = max(np.abs(v).max() for v in line_dists(truth, pix, net_h).values())
    assert worst < 1e-6, f"exact-calib residual {worst}"

    bad = dict(truth, rvec=[1.70, 0.58, 0.02], tvec=[1.9, 2.4, 15.2], f=980.0)
    calib, med = solve_lines(pix, bad, net_h=2.43)
    assert med < 0.5, f"solve did not converge: {med:.2f}px"
    assert abs(calib["net_h_est"] - net_h) < 0.05, calib["net_h_est"]

    # camera in the net plane (x=0): pose still solvable, net height is not
    onaxis = {"img_w": 1280, "img_h": 720, "f": 1100.0,
              "rvec": [1.8, 0.0, 0.0], "tvec": [0.0, 2.0, 16.0], "err": 0.0}
    c2, med2 = solve_lines(_sample(onaxis, net_h), dict(onaxis, f=980.0))
    assert med2 < 0.5, f"on-axis solve failed: {med2:.2f}px"
    assert "net_h_est" not in c2, "unobservable net height reported anyway"

    # net band segmented at a height outside the plausible range (mis-segmented
    # onto a bleacher rail, say) pins the optimizer to its bound — not a reading
    c3, _ = solve_lines(_sample(truth, 3.4), dict(truth, f=980.0))
    assert "net_h_est" not in c3, f"bound-pinned height reported: {c3.get('net_h_est')}"
    print(f"lineseg self-check ok: {med:.4f}px, net_h {calib['net_h_est']}; "
          f"on-axis {med2:.4f}px, net height correctly withheld")


if __name__ == "__main__":
    _self_check()

"""Is this calibration actually right? A metric that cannot be gamed by sparsity.

The trap this exists to close: every error metric we had measured how well the
solver agreed with the evidence it was fitted to, and reported the median over
whatever happened to match. Both reward SPARSITY — fewer, shorter pieces of
evidence fit any pose better. Measured on real footage: the best-scoring clip
of ten (1.34px) was a visibly wrong pose fitted to a handful of fragments,
while a correct calibration scored 1.89px.

So this scores three things, and a calibration has to pass all of them:

  matched   how many of the 8 model lines have supporting image evidence at all
  coverage  what FRACTION of each projected line's length is actually covered
  err       perpendicular distance, over the lines that did match

Coverage is the one that closes the hole. A wrong pose can put a short fragment
near a projected line; it cannot cover 60% of an 18 m sideline's length.

Evidence is a Canny edge map plus a distance transform — deliberately NOT the
segmentation that produced the pose, so this is an independent check rather
than a measure of self-consistency.

Chamfer, not segment matching. The first version of this scored against
lines.detect_segments and failed validation: correct calibrations scored
coverage 0.20-0.35 because the tophat detector only ever recovers a third of
each line, so the metric was measuring DETECTOR recall, not pose correctness.
Asking "is there an edge pixel near this projected line" needs no grouping,
no thresholds on segment length, and no detector to be good.
"""

import cv2
import numpy as np

from .calibrate import project
from .court import ATTACK, HL, HW, NET_H
from .lines import floor_mask

MODEL = {
    # left = +Z = image-left (court.py convention)
    "side_left": ((-HL, 0, HW), (HL, 0, HW)),
    "side_right": ((-HL, 0, -HW), (HL, 0, -HW)),
    "end_far": ((-HL, 0, -HW), (-HL, 0, HW)),
    "end_near": ((HL, 0, -HW), (HL, 0, HW)),
    "attack_far": ((-ATTACK, 0, -HW), (-ATTACK, 0, HW)),
    "attack_near": ((ATTACK, 0, -HW), (ATTACK, 0, HW)),
    "center": ((0, 0, -HW), (0, 0, HW)),
    "net": ((0, NET_H, -HW), (0, NET_H, HW)),
}


def edge_dt(img, lo=50, hi=150):
    """Distance (px) from every pixel to the nearest Canny edge."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), lo, hi)
    return cv2.distanceTransform(255 - e, cv2.DIST_L2, 3)


def line_dt(img, thresh=12.0, min_px=400):
    """Distance to the nearest thing that looks like a PAINTED LINE ON A FLOOR.

    Canny answers "is there an edge here", and in a gym the honest answer is
    yes almost everywhere — ceiling trusses, bleachers, chairs, crowd. Measured
    on real footage, edge_dt scored a visibly correct pose at coverage 0.43 and
    a pose thrown into the ceiling at 0.77, i.e. it ranked them backwards. The
    stripe test (flanks match each other) plus the floor mask throws out
    everything that is not on the playing surface, which is what the coverage
    number was always supposed to mean.
    """
    from .lines import floor_mask, stripe_response

    m = ((stripe_response(img) > thresh).astype(np.uint8) * 255)
    m = cv2.bitwise_and(m, floor_mask(img))
    if int(m.sum()) // 255 < min_px:   # dim or blown-out floor: better than nothing
        return edge_dt(img)
    return cv2.distanceTransform(255 - m, cv2.DIST_L2, 3)


def _support(p, q, dt, near=4.0, step=3.0):
    """Coverage and error for the projected segment p->q against an edge DT.

    Samples along the line and asks how many sample points have an edge within
    `near` px. Points outside the frame are not counted either way — a line
    running off-screen is neither supported nor evidence against.
    """
    L = float(np.linalg.norm(q - p))
    if L < 40:
        return None
    n = max(int(L / step), 8)
    t = np.linspace(0, 1, n)[:, None]
    pts = p + (q - p) * t
    h, w = dt.shape
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if inside.sum() < 8:
        return None
    d = dt[y[inside], x[inside]]
    return float((d <= near).mean()), float(np.median(d)), int(inside.sum())


def _over_mask(p, q, mask, frac=0.5):
    """Does the segment p->q lie over the evidence mask enough to be judged?

    Guards the net line: line_dt only holds evidence on the floor, so a net
    band projected clear of the mask has nothing to be measured against.
    """
    pts = p + (q - p) * np.linspace(0, 1, 60)[:, None]
    h, w = mask.shape
    x = np.round(pts[:, 0]).astype(int)
    y = np.round(pts[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if inside.sum() < 8:
        return False
    return bool((mask[y[inside], x[inside]] > 0).mean() >= frac)


def score(img, calib, net_h=None):
    """Per-line support for a calibration on one frame.

    Returns {line: {"coverage": f, "err": px}} plus aggregate keys.
    net_h shifts the net line to its measured height (regulation if None).
    """
    dt = line_dt(img)
    # The net is the ONLY model line off the floor plane, and line_dt masks to
    # the floor on purpose. So the net can only be judged when the floor mask
    # happens to spill over it -- otherwise there is no evidence there at all,
    # and _support reports that ABSENCE as a huge error. Measured on
    # data/test/test_030: a GT pose whose net band visibly lies along the
    # real net had 0% of that band inside the mask and scored 240px off, while
    # frames where the mask did cover the net scored 0.0px. Same pose quality,
    # opposite verdicts, decided by luck.
    #
    # "No evidence" is not "wrong", so say so: the net comes back None and the
    # caller must treat that as unjudgeable. Scoring it on unmasked Canny edges
    # instead was tried and is worse -- a cluttered hall puts an edge near
    # anything, and a known L/R-flipped pose scored 0.95px.
    fmask = floor_mask(img)
    dy = 0.0 if net_h is None else net_h - NET_H
    out, covs, errs = {}, [], []
    for name, (a, b) in MODEL.items():
        if name == "net":
            a, b = (a[0], a[1] + dy, a[2]), (b[0], b[1] + dy, b[2])
        uv = project(calib, [a, b])
        if not np.isfinite(uv).all():
            out[name] = {"coverage": None, "err": None}
            continue
        if name == "net" and not _over_mask(uv[0], uv[1], fmask):
            out[name] = {"coverage": None, "err": None}
            continue
        r = _support(uv[0], uv[1], dt)
        if r is None:                     # off-screen or too short to judge
            out[name] = {"coverage": None, "err": None}
            continue
        cov, err, npts = r
        out[name] = {"coverage": round(cov, 3), "err": round(err, 2), "n": npts}
        if name != "net":
            covs.append(cov)
            errs.append(err)
    judged = [k for k, v in out.items() if v["coverage"] is not None and k != "net"]
    matched = [k for k in judged if out[k]["coverage"] >= 0.5]
    return {
        "lines": out,
        "n_judged": len(judged),
        "n_matched": len(matched),
        "mean_coverage": round(float(np.mean(covs)), 3) if covs else 0.0,
        "err": round(float(np.median(errs)), 2) if errs else None,
        # Surfaced separately because it is the sharpest signal we have and the
        # aggregates above deliberately exclude it. The net is the only
        # high-contrast structure OFF the floor, so a wrong pose cannot put it
        # in the right place, while floor lines are ambiguous enough that even a
        # collapsed wireframe lands near something. Measured on three calibs
        # with known ground truth: correct -> net err 1.9px / coverage 1.00;
        # two wrong -> 262.5px and 135.3px, both coverage 0.00. Floor-only
        # mean coverage ranked those BACKWARDS (correct 0.33, wrong 0.37).
        "net_err": out["net"]["err"],
        "net_coverage": out["net"]["coverage"],
    }


def net_ok(s, max_err=30.0):
    """Is the NET where the pose says it is? (ok, reason).

    Cheap, sharp, and independent of the floor-coverage thresholds, which do
    not separate good poses from bad on glossy-wood gyms. Use this to gate a
    calibration when floor coverage is inconclusive. None when the net does not
    project into frame — absence of evidence, not evidence of a bad pose.
    """
    e = s.get("net_err")
    if e is None:
        return None, "net not in frame"
    if e > max_err:
        return False, f"net {e:.0f}px from the real net band (>{max_err:.0f})"
    return True, f"net matches at {e:.1f}px"


def verdict(s, min_matched=4, min_coverage=0.6, max_err=5.0):
    """(ok, reasons). Every gate must pass — that is the whole point.

    A pose is only accepted when enough DISTINCT model lines are supported over
    enough of their LENGTH and sit close to the evidence. Dropping any one of
    the three re-opens the sparsity hole.
    """
    reasons = []
    if s["n_matched"] < min_matched:
        reasons.append(f"only {s['n_matched']}/{s['n_judged']} court lines supported "
                       f"(need {min_matched})")
    if s["mean_coverage"] < min_coverage:
        reasons.append(f"mean coverage {s['mean_coverage']:.2f} < {min_coverage} "
                       f"— lines projected where there is no paint")
    if s["err"] is None:
        reasons.append("no line had enough support to measure an error")
    elif s["err"] > max_err:
        reasons.append(f"median err {s['err']:.1f}px > {max_err}")
    return (not reasons), reasons


def _self_check():
    """A right pose passes; a wrong one fails — on the SAME synthetic image."""
    import cv2

    truth = {"img_w": 1280, "img_h": 720, "f": 1100.0,
             "rvec": [1.75, 0.55, 0.05], "tvec": [1.5, 2.0, 16.0], "err": 0.0}
    img = np.full((720, 1280, 3), (70, 100, 150), np.uint8)  # wood-ish floor
    for a, b in MODEL.values():
        uv = project(truth, np.linspace(a, b, 60))
        ok = np.isfinite(uv).all(1)
        for i in range(len(uv) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(img, tuple(uv[i].astype(int)), tuple(uv[i + 1].astype(int)),
                         (245, 245, 245), 4)

    good = score(img, truth)
    ok_g, why_g = verdict(good)
    assert ok_g, f"true pose rejected: {why_g} ({good})"

    # a pose wrong by ~a metre of camera position: lines land off the paint
    bad = dict(truth, tvec=[1.5, 2.0, 13.0], rvec=[1.62, 0.55, 0.05])
    s_bad = score(img, bad)
    ok_b, why_b = verdict(s_bad)
    assert not ok_b, f"wrong pose accepted: {s_bad}"

    # the sparsity trap itself: erase all but one line. Old metrics loved this
    # (few matches, tiny median). Coverage must refuse it.
    sparse = np.full((720, 1280, 3), (70, 100, 150), np.uint8)
    a, b = MODEL["end_near"]
    uv = project(truth, np.linspace(a, b, 60))
    for i in range(len(uv) - 1):
        cv2.line(sparse, tuple(uv[i].astype(int)), tuple(uv[i + 1].astype(int)),
                 (245, 245, 245), 4)
    ok_s, why_s = verdict(score(sparse, truth))
    assert not ok_s, "one supported line was enough to pass — sparsity hole open"

    print(f"calib_score self-check ok: true pose passes "
          f"({good['n_matched']}/{good['n_judged']} lines, "
          f"coverage {good['mean_coverage']:.2f}, err {good['err']}px); "
          f"wrong pose fails ({why_b[0]}); sparse evidence fails ({why_s[0]})")


if __name__ == "__main__":
    _self_check()

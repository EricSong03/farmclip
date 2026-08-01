"""Find the court by searching camera poses, not by classifying pixels.

Bottom-up calibration (segment the lines, name them, fit a pose) has to
re-learn what a sideline looks like at every venue. This inverts it: the court
is a rigid body of exactly known size, so ask which camera pose puts the known
18x9 m model onto this image's line evidence, and search for it directly.

Nothing here is trained. The only model is the FIVB rulebook, so a venue the
segmenter has never seen costs nothing extra.

Why it is tractable: the pose manifold is 6-D once you use the domain
constraints (camera behind an end line, looking at the court, level horizon),
and one hypothesis costs a projection plus a table lookup. Cross-entropy
method with restarts converges in ~10^6 evaluations, i.e. seconds.

Two deliberate choices:

- **Off-frame samples are excluded from the error but gated on separately.**
  Scoring them as maximum error looks like the safe choice and is not: near
  corners are genuinely out of frame in this footage (that is the premise of
  the whole repo), so it scores the TRUE pose at 11.9/20 and hands the win to
  a pose that crams the court onscreen. Measured, not theorised. The sparsity
  hole it was meant to close — push the court off-screen until only a
  well-fitting fragment is left — is closed by the two gates instead: enough
  of the model must be in frame, and it must SPAN the frame.
- **Evidence is the stripe response (lines.stripe_response), not Canny.** A
  painted line has matching floor on both flanks; a chair edge, banner border
  or jersey seam does not. Canny cannot tell them apart and the search will
  happily fit the clutter.
"""

import cv2
import numpy as np

from .court import ATTACK, HL, HW, NET_H

# (lo, hi) per parameter: camera xyz, aim xz on the floor, focal / image width.
# Camera restricted to +X: the court is symmetric under 180 deg about Y, so the
# -X half of the space holds only duplicates of poses already searched.
BOUNDS = np.array([
    (7.0, 32.0),    # cam x — behind the near end line (x=+9), give or take
    (0.8, 12.0),    # cam y — tripod to balcony
    (-14.0, 14.0),  # cam z — lateral offset
    (-9.0, 9.0),    # aim x — a point on the floor the camera looks at
    (-7.0, 7.0),    # aim z
    (0.4, 3.0),     # focal, in image widths
])

MODEL_LINES = [
    ((-HL, 0, -HW), (+HL, 0, -HW)),
    ((-HL, 0, +HW), (+HL, 0, +HW)),
    ((-HL, 0, -HW), (-HL, 0, +HW)),
    ((+HL, 0, -HW), (+HL, 0, +HW)),
    ((-ATTACK, 0, -HW), (-ATTACK, 0, +HW)),
    ((+ATTACK, 0, -HW), (+ATTACK, 0, +HW)),
    ((0, 0, -HW), (0, 0, +HW)),
    ((0, NET_H, -HW), (0, NET_H, +HW)),
]


def model_points(per_line: int = 12) -> np.ndarray:
    """(N,3) samples spread along every model line — equal count per line, so a
    18 m sideline cannot outvote a 9 m attack line."""
    return np.vstack([np.linspace(a, b, per_line) for a, b in MODEL_LINES])


def cost_map(frame, thresh: float = 12.0, min_px: int = 400) -> np.ndarray:
    """Pixel -> distance to the nearest thing that looks like a painted line.

    Stripe response first (see module docstring); falls back to Canny when the
    floor is too dark/washed out for the stripe test to fire at all, because a
    weak cost map beats no cost map.
    """
    mask = (lines_stripe(frame) > thresh).astype(np.uint8) * 255
    if int(mask.sum()) // 255 < min_px:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150)
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


def lines_stripe(frame):
    from .lines import stripe_response  # deferred: lines imports are heavy
    return stripe_response(frame)


def poses(p: np.ndarray):
    """(H,6) params -> (R (H,3,3) world->camera, t (H,3), f (H,)).

    Look-at parameterization with a level horizon: every sample is a camera
    pointed at a point ON THE COURT, which is what makes 6 dimensions enough.
    Free rvec would spend most of its volume aimed at the ceiling.
    """
    C = p[:, :3]
    A = np.stack([p[:, 3], np.zeros(len(p)), p[:, 4]], 1)
    fwd = A - C
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    xc = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    xc /= np.linalg.norm(xc, axis=1, keepdims=True)
    yc = np.cross(fwd, xc)
    R = np.stack([xc, yc, fwd], axis=1)
    t = -np.einsum("hij,hj->hi", R, C)
    return R, t, p[:, 5]


REJECT = 1e6      # finite, so argsort stays well-defined when a batch is all bad
MIN_VISIBLE = 0.5  # fraction of model samples that must land in frame
MIN_SPAN = (0.30, 0.12)  # projected court must cover this much of (width, height)


def scores(p: np.ndarray, pts: np.ndarray, dt: np.ndarray, w: int, h: int,
           max_d: float | None = None, chunk: int = 4096) -> np.ndarray:
    """Mean clipped distance from each hypothesis' projected court to the
    evidence, over the samples that are actually in frame. Lower is better.

    Two gates keep "in frame only" from being gameable (see module docstring):
    enough of the model must be visible at all, and what is visible must span
    the frame — a court collapsed to a blob sitting on one painted line
    otherwise scores a perfect zero.
    """
    max_d = w / 64.0 if max_d is None else max_d
    out = np.empty(len(p), np.float32)
    for i in range(0, len(p), chunk):
        q = p[i:i + chunk]
        R, t, f = poses(q)
        fpx = (f * w)[:, None]  # param 6 is focal / image width, not pixels
        cam = np.einsum("hij,mj->hmi", R, pts) + t[:, None, :]
        z = cam[:, :, 2]
        ok = z > 0.1
        zs = np.where(ok, z, 1.0)
        u = fpx * cam[:, :, 0] / zs + w / 2
        v = fpx * cam[:, :, 1] / zs + h / 2
        ui = np.clip(u, 0, w - 1).astype(np.int32)
        vi = np.clip(v, 0, h - 1).astype(np.int32)
        inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        n_in = inside.sum(1)
        d = np.where(inside, np.minimum(dt[vi, ui], max_d), 0.0)
        s = d.sum(1) / np.maximum(n_in, 1)
        uin = np.where(inside, u, np.nan)
        vin = np.where(inside, v, np.nan)
        with np.errstate(invalid="ignore"):
            span_u = np.nanmax(uin, axis=1) - np.nanmin(uin, axis=1)
            span_v = np.nanmax(vin, axis=1) - np.nanmin(vin, axis=1)
        bad = ((n_in < MIN_VISIBLE * pts.shape[0])
               | ~np.isfinite(span_u) | ~np.isfinite(span_v)
               | (span_u < MIN_SPAN[0] * w) | (span_v < MIN_SPAN[1] * h))
        out[i:i + chunk] = np.where(bad, REJECT, s)
    return out


def search(dt: np.ndarray, w: int, h: int, restarts: int = 6, iters: int = 9,
           n: int = 20000, elite: float = 0.02, per_line: int = 12,
           seed: int = 0, bounds: np.ndarray | None = None) -> dict:
    """Cross-entropy search for the pose that best explains the evidence map.

    CEM rather than a grid: 6-D grids are hopeless at useful resolution, and
    CEM spends its samples where the score is already good. Restarts because a
    single run collapses into whichever basin it found first, and a court has
    plausible-looking wrong basins (the model fitted to one half of the floor,
    or to an adjacent court's lines).

    The distance clip is ANNEALED, wide to narrow, and that is what makes the
    search work at all. At the final 20 px clip the correct optimum occupies
    ~1e-7 of the search volume, so no amount of sampling finds it — measured:
    every restart converged to the same flat wrong basin. A wide clip blurs the
    objective until poses that are merely in the neighbourhood already score
    better than garbage, which is a basin CEM can actually climb; sharpening it
    each iteration then walks that neighbourhood onto the paint. Standard
    coarse-to-fine chamfer matching.
    """
    b = BOUNDS if bounds is None else bounds
    pts = model_points(per_line)
    rng = np.random.default_rng(seed)
    n_elite = max(int(n * elite), 8)
    d0, d1 = w / 6.0, w / 64.0
    best = (np.inf, None)
    for r in range(restarts):
        mu = rng.uniform(b[:, 0], b[:, 1])
        sd = (b[:, 1] - b[:, 0]) / 3.0
        for k in range(iters):
            md = d0 * (d1 / d0) ** (k / max(iters - 1, 1))
            p = rng.normal(mu, sd, size=(n, 6))
            np.clip(p, b[:, 0], b[:, 1], out=p)
            s = scores(p, pts, dt, w, h, max_d=md)
            top = p[np.argsort(s)[:n_elite]]
            mu, sd = top.mean(0), top.std(0)
            # variance floor: without it the first good basin collapses sd to
            # zero in ~3 iterations and the remaining samples are all the same
            # point. Costs nothing, buys the local refinement.
            sd = np.maximum(sd, (b[:, 1] - b[:, 0]) * 1e-3)
        s0 = float(scores(mu[None], pts, dt, w, h)[0])
        if s0 < best[0]:
            best = (s0, mu.copy())
    return to_calib(best[1], w, h, best[0])


def params_from_calib(calib: dict) -> np.ndarray:
    """calib dict -> the 6 search parameters (inverse of to_calib)."""
    R = cv2.Rodrigues(np.array(calib["rvec"], float))[0]
    t = np.array(calib["tvec"], float).reshape(3)
    C = -R.T @ t
    fwd = R.T @ np.array([0.0, 0.0, 1.0])   # optical axis in world coords
    # where the axis meets the floor; a camera aimed level or upward has no
    # intersection, so fall back to aiming at the court centre
    k = -C[1] / fwd[1] if abs(fwd[1]) > 1e-6 and fwd[1] < 0 else None
    aim = C + k * fwd if k is not None and k > 0 else np.zeros(3)
    return np.array([C[0], C[1], C[2], aim[0], aim[2],
                     calib["f"] / calib["img_w"]], float)


# Half-width of the local search box, per parameter. Sized from the measured
# capture radius, not guessed: the objective climbs ~5px per metre of camera
# position, so a box much wider than this just re-opens the global problem.
REFINE_SPAN = np.array([1.5, 0.8, 1.5, 2.0, 2.0, 0.15])


def refine(dt: np.ndarray, w: int, h: int, init: dict,
           span: np.ndarray | None = None, restarts: int = 2, iters: int = 8,
           n: int = 8000, seed: int = 0, guard_img=None) -> dict:
    """Pull an existing calib onto the line evidence.

    This is the half of the search that works. It needs a seed inside the
    capture radius, which the keypoint or lineseg calib supplies; what it adds
    is an objective built from evidence NEITHER of those models produced, so it
    can correct a pose that a segmenter was confidently wrong about.

    guard_img: keep the refined pose only if it does not lose ground on
    calib_score, which judges by edge COVERAGE rather than mean distance.
    Without it this wrecks poses that were already right — measured 2 of 13 on
    the benchmark clips, because minimising mean distance will happily slide a
    correct court sideways onto a denser patch of another sport's markings.
    Coverage is the term that notices; the search objective has no equivalent.
    """
    p0 = params_from_calib(init)
    sp = REFINE_SPAN if span is None else span
    lo, hi = p0 - sp, p0 + sp
    # Intersect with the global box where that leaves a box at all. A seed can
    # legitimately sit outside it — a camera on the -X side (BOUNDS searches
    # only +X, the court being 180deg-symmetric), or a pose the keypoint solve
    # put underground — and clamping then inverts the range.
    clo, chi = np.maximum(lo, BOUNDS[:, 0]), np.minimum(hi, BOUNDS[:, 1])
    empty = clo >= chi
    clo[empty], chi[empty] = lo[empty], hi[empty]
    out = search(dt, w, h, restarts=restarts, iters=iters, n=n, seed=seed,
                 bounds=np.stack([clo, chi], axis=1))
    base = float(scores(p0[None], model_points(12), dt, w, h)[0])
    if out["err"] >= base:
        return dict(init, err=base, method="search-kept-init")
    if guard_img is not None:
        from .calib_score import score as cscore
        a = cscore(guard_img, init, init.get("net_h_est"))
        b = cscore(guard_img, out, out.get("net_h_est"))
        if (b["n_matched"], b["mean_coverage"]) < (a["n_matched"], a["mean_coverage"]):
            return dict(init, method="search-kept-init")
    return out


def to_calib(p: np.ndarray, w: int, h: int, err: float) -> dict:
    R, t, f = poses(p[None])
    rvec = cv2.Rodrigues(R[0])[0].ravel()
    return {"img_w": w, "img_h": h, "f": float(f[0] * w),
            "rvec": rvec.tolist(), "tvec": t[0].tolist(),
            "err": float(err), "per_point_err": {}, "method": "search",
            "cam_xyz": [round(float(v), 2) for v in p[:3]]}


def _render(calib, size=(720, 1280)):
    """A synthetic gym: wood floor, white court lines, one distractor line."""
    from .calibrate import project
    img = np.full((*size, 3), (70, 100, 150), np.uint8)
    for a, b in MODEL_LINES:
        uv = project(calib, np.linspace(a, b, 60))
        ok = np.isfinite(uv).all(1)
        for i in range(len(uv) - 1):
            if ok[i] and ok[i + 1]:
                cv2.line(img, tuple(uv[i].astype(int)), tuple(uv[i + 1].astype(int)),
                         (240, 240, 240), 4)
    cv2.line(img, (0, 700), (1279, 640), (235, 235, 235), 4)  # badminton-ish
    return img


def _self_check():
    """The objective is exact on a known pose, and refine() climbs to it."""
    from .calibrate import project
    from .court import KEYPOINTS

    p_true = np.array([16.0, 3.2, 2.0, -1.0, 0.5, 1.05])
    truth = to_calib(p_true, 1280, 720, 0.0)
    img = _render(truth)
    dt = cost_map(img)

    s = float(scores(p_true[None], model_points(12), dt, 1280, 720)[0])
    assert s < 0.5, f"true pose does not score ~0 on its own render: {s}"

    obj = np.array(list(KEYPOINTS.values()), float)
    ref = project(truth, obj)

    def kp_err(c):
        b = project(c, obj)
        m = np.isfinite(ref).all(1) & np.isfinite(b).all(1)
        return float(np.median(np.linalg.norm(ref[m] - b[m], axis=1)))

    # a seed off by ~50px of keypoint error, i.e. a plausible sloppy calib
    seed = to_calib(p_true + np.array([0.6, 0.3, -0.7, 0.8, -0.9, 0.05]),
                    1280, 720, 0.0)
    before = kp_err(seed)
    after = kp_err(refine(dt, 1280, 720, seed, seed=1))
    assert after < 5.0, f"refine did not converge: {before:.0f} -> {after:.0f}px"
    print(f"court_search self-check ok: exact pose scores {s:.3f}px; "
          f"refine pulled a {before:.0f}px seed to {after:.1f}px")


# ponytail: search() is deliberately kept even though blind global search does
# NOT solve real frames — measured, see docs/plans/court-search.md. It is what
# refine() runs inside a small box, and it is the honest place to plug a
# correspondence-based initializer (vanishing points) when that exists.


if __name__ == "__main__":
    _self_check()

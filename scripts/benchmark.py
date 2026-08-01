"""Score calibration methods as POSE ERROR against hand-labelled ground truth.

Every proxy this repo has invented has, at some point, passed a visibly wrong
pose: reprojection error ranked a wrong pose (1.34px) above a correct one
(1.89px); line coverage ranked a correct calib below a wrong one; the
court_search objective scored a court collapsed onto clutter at 0.55px. They
share one flaw -- each asks "is there evidence near the projected model?",
which is trivially satisfied by sliding or shrinking the court onto any dense
clutter. None asks whether the evidence is EXPLAINED.

out/testimgs fixes that: 19 hand-clicked frames from 11 VODs with zero overlap
against the 29 training VODs. solve_labeled on those clicks is a known-correct
pose, so a method can finally be scored against truth instead of against
another proxy.

Two numbers per frame, because one is not enough:

- kp_px  -- median pixel distance between court keypoints projected under the
            candidate and under ground truth. What you'd see in an overlay.
- floor_m -- back-project a grid of image points onto the floor under both
            poses and take the median distance IN METRES. Pixel agreement
            alone hides the focal/height ambiguity: a pose can sit on the floor
            lines at 2px and still put the camera metres off, which silently
            mirrors or rescales every world coordinate downstream.

Usage:
    python -m uv run python scripts/benchmark.py
    python -m uv run python scripts/benchmark.py --methods v6b,search --limit 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from farmclip.calibrate import draw_overlay, project, solve_labeled  # noqa: E402
from farmclip.court import HL, HW, KEYPOINTS, NET_H, depth_span, width_span  # noqa: E402
from farmclip.track import pixel_to_floor  # noqa: E402

TEST = Path("out/testimgs")
BENCH = Path("out/bench")

# A ground-truth pose that is itself doubtful is worse than no ground truth:
# it would score correct methods as wrong. Frames failing these get reported
# and excluded, not silently fudged.
GT_MAX_ERR = 15.0

# Verdict thresholds. Deliberately loose -- this is "usable calibration", not
# "good one". A method that cannot clear these is not competitive.
PASS_PX, PASS_M = 20.0, 1.0

OBJ = np.array(list(KEYPOINTS.values()), float)
NET_BAND = np.linspace((0, NET_H, -HW), (0, NET_H, +HW), 11)


def gt_reject(clicks: dict, gt: dict, frame) -> str | None:
    """Why this labelling cannot serve as ground truth, or None if it can.

    A low click residual is NOT evidence of a correct pose, and assuming it was
    is what this whole file exists to stop. Measured on this very set:
    test_000 fits its 6 clicks at 1.9px and projects a court that misses the
    paint entirely, because all 6 clicks sit in ONE vertical plane (z = -HW) --
    a configuration PnP cannot pin down, so the solver is free to pick any of a
    family of poses and reports a tiny residual for whichever it lands on.

    So gate on three independent things:
      1. is the pose physical (camera above the floor, residual sane)
      2. do the clicks actually CONSTRAIN a pose (two depth planes, real span)
      3. does the resulting court land on the painted lines, and specifically
         on the NET -- the only high-contrast structure off the floor plane, so
         it is the one thing a floor-fitted-but-wrong-focal pose cannot fake

    Check 3 leans on calib_score, which is a one-way proxy and can pass a wrong
    pose. That is fine HERE and nowhere else: it is used only to veto. When it
    reports the net 80px off, the pose is wrong -- that direction is trustworthy.
    """
    if gt.get("cam_below_floor"):
        return "camera below the floor"
    if gt["err"] > GT_MAX_ERR:
        return f"click residual {gt['err']:.1f}px > {GT_MAX_ERR}px"
    if width_span(clicks) < 4.0:
        return "degenerate: every click in one vertical plane (pose unconstrained)"
    if depth_span(clicks) < 9.0:
        return f"depth span only {depth_span(clicks):.0f}m (focal/distance trade off)"
    # Check 3 VETOES a clear failure; it does not demand a clear success. That
    # asymmetry is deliberate and was learned the hard way: requiring
    # n_matched >= 4 rejected test_016 and test_021, whose overlays are both
    # visibly correct, because a legitimately off-frame sideline leaves too few
    # lines judgeable. A one-way proxy can be trusted when it says "wrong" and
    # not when it says "right", so it is only allowed to say "wrong".
    from farmclip.calib_score import score
    s = score(frame, gt, gt.get("net_h_est"))
    net = s["lines"]["net"]
    # net err None = the band lies clear of the floor evidence mask, so there
    # is nothing to measure it against. Unjudgeable is not failure; the
    # coverage check below still has to pass either way.
    if net["err"] is not None and net["err"] > 15.0:
        return f"net {net['err']}px off the band (floor fits, height/focal wrong)"
    # 0.50 is calibrated against visual inspection, not chosen a priori:
    # test_016 (0.55) and test_021 (0.72) are confirmed correct by eye and must
    # survive; test_014 (0.44) is confirmed wrong by eye and must not. If you
    # move this threshold, open the overlays it changes and check them.
    if s["mean_coverage"] < 0.50:
        return f"court misses the paint (coverage {s['mean_coverage']:.2f})"
    return None


def ground_truth(limit=None):
    """(name, frame, gt_calib, clicks) per usable test frame. Prints rejects."""
    out, rejects = [], []
    for lp in sorted((TEST / "labels").glob("*.json")):
        img = TEST / (lp.stem + ".jpg")
        if not img.exists():
            continue
        frame = cv2.imread(str(img))
        h, w = frame.shape[:2]
        clicks = {k: v for k, v in json.loads(lp.read_text()).items()
                  if not k.startswith("post_")}
        gt = solve_labeled(clicks, w, h)
        why = gt_reject(clicks, gt, frame)
        if why:
            rejects.append((lp.stem, why))
            continue
        out.append((lp.stem, frame, gt, clicks))
        if limit and len(out) >= limit:
            break
    for name, why in rejects:
        print(f"  [reject] {name}: {why}")
    return out


def _in_frame(uv, w, h, margin=0.25):
    """Points inside the image, plus a margin -- most of this court is
    legitimately off-screen, so scoring only strictly-visible points throws
    away the far end where focal error shows up most."""
    mx, my = w * margin, h * margin
    return (np.isfinite(uv).all(1) & (uv[:, 0] > -mx) & (uv[:, 0] < w + mx)
            & (uv[:, 1] > -my) & (uv[:, 1] < h + my))


def compare(gt: dict, cand: dict, w: int, h: int) -> dict:
    """Candidate pose vs ground-truth pose. Pixels AND metres."""
    a, b = project(gt, OBJ), project(cand, OBJ)
    m = _in_frame(a, w, h) & np.isfinite(b).all(1)
    kp_px = float(np.median(np.linalg.norm(a[m] - b[m], axis=1))) if m.sum() >= 3 else float("inf")

    # net band gets its own number: it is the only evidence off the floor
    # plane, so it is where a pose that fits the floor at 2px and has the
    # camera height wrong finally shows the error.
    na, nb = project(gt, NET_BAND), project(cand, NET_BAND)
    nm = np.isfinite(na).all(1) & np.isfinite(nb).all(1)
    net_px = float(np.median(np.linalg.norm(na[nm] - nb[nm], axis=1))) if nm.sum() >= 3 else float("inf")

    # Metric error: where does each pose think a given pixel lands on the
    # floor? Sampled over the lower two-thirds of the frame (above that the
    # rays graze the horizon and the metres explode without meaning anything).
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 7) * w, np.linspace(0.35, 0.95, 5) * h)
    uv = np.stack([gx.ravel(), gy.ravel()], 1)
    pa, pb = pixel_to_floor(gt, uv), pixel_to_floor(cand, uv)
    keep = (np.isfinite(pa).all(1) & np.isfinite(pb).all(1)
            & (np.abs(pa[:, 0]) < HL + 8) & (np.abs(pa[:, 2]) < HW + 8))
    floor_m = float(np.median(np.linalg.norm(pa[keep] - pb[keep], axis=1))) if keep.sum() >= 3 else float("inf")

    def cam(c):
        R = cv2.Rodrigues(np.array(c["rvec"], float).reshape(3, 1))[0]
        return (-R.T @ np.array(c["tvec"], float).reshape(3, 1)).ravel()

    return {"kp_px": kp_px, "net_px": net_px, "floor_m": floor_m,
            "cam_m": float(np.linalg.norm(cam(gt) - cam(cand))),
            "f_ratio": float(cand["f"] / gt["f"]),
            "pass": bool(kp_px <= PASS_PX and floor_m <= PASS_M)}


# --- methods: (frame, w, h, ctx) -> calib or None ---------------------------

def m_v6b(frame, w, h, ctx):
    from farmclip.kp_detect import detect_keypoints
    kp = detect_keypoints(frame)
    ctx["n_kp"] = len(kp)
    if len(kp) < 4:
        return None
    c = solve_labeled(kp, w, h)
    ctx["kp_reproj"] = round(c["err"], 1)
    ctx["cam_below_floor"] = bool(c.get("cam_below_floor"))
    return c


def m_lineseg(frame, w, h, ctx):
    """Needs an init pose -- lineseg refines, it does not find. v6b supplies
    it, which is exactly how cli.calibrate chains them, so a v6b failure is
    honestly a lineseg failure too."""
    from farmclip.lineseg import detect_lines, solve_lines
    init = ctx.get("v6b_calib")
    if init is None:
        ctx["why"] = "no v6b init"
        return None
    pix = detect_lines([frame])
    ctx["n_classes"] = len(pix)
    if len(pix) < 3:
        return None
    c, err = solve_lines(pix, init)
    ctx["lineseg_err"] = round(float(err), 1)
    return c


def m_classical(frame, w, h, ctx):
    from farmclip.hypothesis import solve_frame
    c, info = solve_frame(frame, w, h)
    if info:
        ctx["n_inliers"] = info.get("n_inliers")
    return c


def m_search(frame, w, h, ctx):
    from farmclip.court_search import cost_map, search
    c = search(cost_map(frame), w, h)
    ctx["search_obj"] = round(c["err"], 2)
    return c


METHODS = {"v6b": m_v6b, "lineseg": m_lineseg, "classical": m_classical,
           "search": m_search}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="v6b,lineseg,classical,search")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-overlays", action="store_true")
    args = ap.parse_args()
    names = [n for n in args.methods.split(",") if n]
    bad = [n for n in names if n not in METHODS]
    if bad:
        sys.exit(f"unknown method(s): {bad}; have {list(METHODS)}")

    print("ground truth:")
    frames = ground_truth(args.limit or None)
    print(f"  {len(frames)} usable frames "
          f"(median {np.median([g['err'] for _, _, g, _ in frames]):.1f}px)")
    if not frames:
        sys.exit("no usable ground truth -- relabel out/testimgs first")

    if not args.no_overlays:
        (BENCH / "gt").mkdir(parents=True, exist_ok=True)
        for name, frame, gt, clicks in frames:
            cv2.imwrite(str(BENCH / "gt" / f"{name}.jpg"),
                        draw_overlay(frame, gt, clicks))

    rows, per_frame = {}, {}
    for name, frame, gt, clicks in frames:
        h, w = frame.shape[:2]
        ctx = {}
        for mn in names:
            t0 = time.time()
            try:
                cand = METHODS[mn](frame, w, h, ctx)
            except Exception as e:                       # a crash is a failure
                cand, ctx["error"] = None, f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            if mn == "v6b":
                ctx["v6b_calib"] = cand
            rec = {"frame": name, "method": mn, "secs": round(dt, 1),
                   **{k: v for k, v in ctx.items() if k != "v6b_calib"}}
            if cand is None:
                rec.update({"pass": False, "kp_px": None})
            else:
                rec.update(compare(gt, cand, w, h))
                if not args.no_overlays:
                    d = BENCH / mn
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{name}.jpg"),
                                draw_overlay(frame, cand, clicks))
            rows.setdefault(mn, []).append(rec)
            per_frame.setdefault(name, {})[mn] = rec
            v = f"{rec['kp_px']:7.1f}px {rec['floor_m']:5.2f}m" if rec.get("kp_px") is not None else "     FAILED    "
            print(f"  {name} {mn:<9} {v} {'PASS' if rec['pass'] else 'fail'} ({dt:.1f}s)")

    print(f"\n{'method':<10} {'pass':>7} {'kp_px':>8} {'net_px':>8} "
          f"{'floor_m':>8} {'cam_m':>7} {'f_ratio':>8} {'secs':>6}")

    def med(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None and np.isfinite(r[k])]
        return np.median(v) if v else float("nan")

    summary = {}
    for mn in names:
        rs = rows[mn]
        ok = [r for r in rs if r["pass"]]
        s = {"pass": f"{len(ok)}/{len(rs)}",
             "kp_px": med(rs, "kp_px"), "net_px": med(rs, "net_px"),
             "floor_m": med(rs, "floor_m"), "cam_m": med(rs, "cam_m"),
             "f_ratio": med(rs, "f_ratio"), "secs": med(rs, "secs")}
        summary[mn] = s
        print(f"{mn:<10} {s['pass']:>7} {s['kp_px']:8.1f} {s['net_px']:8.1f} "
              f"{s['floor_m']:8.2f} {s['cam_m']:7.2f} {s['f_ratio']:8.2f} {s['secs']:6.1f}")
    print("\nmedians are over frames the method PRODUCED a pose for; the pass "
          "column is over all frames, so a method that answers rarely but well "
          "looks good on the medians and bad on the pass count. Read both.")

    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "results.json").write_text(json.dumps(
        {"thresholds": {"kp_px": PASS_PX, "floor_m": PASS_M,
                        "gt_max_err": GT_MAX_ERR},
         "summary": summary, "per_frame": per_frame}, indent=1, default=str))
    print(f"-> {BENCH / 'results.json'}, overlays in {BENCH}/<method>/")


def _self_check():
    """A pose compared against itself is zero; a mirrored one is not.

    The mirror case is the point: it projects the symmetric court to nearly the
    same pixels, which is how every previous metric was fooled. compare() must
    catch it in metres even when pixels look fine."""
    gt = solve_labeled({n: p for n, p in zip(
        KEYPOINTS, project({"img_w": 1280, "img_h": 720, "f": 1300.0,
                            "rvec": [0.1, 0.05, 0.0], "tvec": [0.0, 2.0, 20.0]},
                           OBJ))}, 1280, 720)
    same = compare(gt, gt, 1280, 720)
    assert same["kp_px"] < 1e-6 and same["floor_m"] < 1e-6, same
    mirror = dict(gt)
    R = cv2.Rodrigues(np.array(gt["rvec"], float).reshape(3, 1))[0]
    M = np.diag([1.0, 1.0, -1.0])                       # flip world Z
    mirror["rvec"] = cv2.Rodrigues(R @ M)[0].ravel().tolist()
    got = compare(gt, mirror, 1280, 720)
    assert not got["pass"], f"mirrored pose passed: {got}"
    assert got["floor_m"] > 1.0, f"mirror invisible in metres: {got}"
    print(f"benchmark self-check ok: identical pose 0.0px/0.00m; "
          f"Z-mirrored pose {got['kp_px']:.0f}px / {got['floor_m']:.1f}m -> fail")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()

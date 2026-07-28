"""Refine calibration: RANSAC over line correspondences, scorer as judge.

Multi-sport floors are full of distractor lines, so any single association
set may be poisoned — sample subsets, ICP-refine each, keep whatever scores
best on cached detections. Usage: python scripts/calib_refine.py <outdir> <video>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.line_icp import refine
from farmclip.lines import detect_segments
from calib_eval import MODEL, match_segs

MISS = 60.0  # px penalty per expected-but-unmatched line slot


def build_seg_cache(video, n_stamps=6):
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cache = []
    for fr in np.linspace(0, n - 1, n_stamps).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ok, img = cap.read()
        if ok:
            cache.append(detect_segments(img))
    cap.release()
    return cache


def score(c, seg_cache):
    court, net = [], []
    for segs in seg_cache:
        m = match_segs(segs, c)
        for k in MODEL:
            e = m[k][0] if k in m else MISS  # dropping lines must not pay
            (net if k == "net" else court).append(e)
    med_c = float(np.mean(np.minimum(court, MISS)))
    med_n = float(np.mean(np.minimum(net, MISS)))
    return med_c + med_n, med_c, med_n


def associations(c, seg_cache, perp_max):
    pairs = []
    for segs in seg_cache:
        for name, (e, seg) in match_segs(segs, c, perp_max=perp_max).items():
            pairs.append((seg, MODEL[name]))
            if name == "net":       # the band detector is the one thing we
                pairs += [(seg, MODEL[name])] * 2  # trust at pixel level
    return pairs


def refine_ransac(calib0, seg_cache, f_anchor, trials=60, seed=0):
    """Returns (best_calib, best_score_tuple). Never returns worse than calib0."""
    rng = np.random.default_rng(seed)
    best_calib, best = calib0, score(calib0, seg_cache)
    f_scale0 = f_anchor / calib0["img_w"]
    for trial in range(trials):
        perp = 90 if trial < trials // 2 else 40
        pairs = associations(best_calib, seg_cache, perp)
        if len(pairs) < 6:
            break
        take = rng.random(len(pairs)) < 0.7
        subset = [p for p, t in zip(pairs, take) if t]
        if len(subset) < 6:
            continue
        try:
            cand = refine(dict(best_calib), subset, iters=3,
                          f_range=(f_scale0 * 0.8, f_scale0 * 1.25))
        except Exception:
            continue
        cand["img_w"], cand["img_h"] = calib0["img_w"], calib0["img_h"]
        s = score(cand, seg_cache)
        if s[0] < best[0] - 0.05:
            best_calib, best = cand, s
    return best_calib, best


if __name__ == "__main__":
    outdir, video = Path(sys.argv[1]), sys.argv[2]
    calib0 = json.loads((outdir / "calib.json").read_text())
    orig = outdir / "calib_orig.json"
    if not orig.exists():
        orig.write_text(json.dumps(calib0, indent=1))
    f_anchor = json.loads(orig.read_text())["f"]
    seg_cache = build_seg_cache(video)
    b = score(calib0, seg_cache)
    print(f"baseline: court {b[1]:.1f}px net {b[2]:.1f}px")
    best_calib, best = refine_ransac(calib0, seg_cache, f_anchor)
    if best_calib is not calib0:
        (outdir / "calib_old.json").write_text(json.dumps(calib0, indent=1))
        (outdir / "calib.json").write_text(json.dumps(
            {k: v for k, v in best_calib.items() if not str(k).startswith("_")},
            indent=1))
        print(f"ACCEPTED: court {best[1]:.1f}px net {best[2]:.1f}px -> calib.json")
    else:
        print("no improvement; calib.json unchanged")

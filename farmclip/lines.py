"""Court-line segment detection for calibration assistance.

Two detectors, two definitions of "line":

- detect_segments (old): a line is anything locally BRIGHT and thin (tophat).
  Cheap, but a chair edge, jersey seam and banner border all qualify — on
  cluttered venues it returns 100+ segments, most of them not court lines.
- detect_stripes (new): a line is a narrow band of near-uniform colour whose
  two FLANKS MATCH EACH OTHER and differ from the band. A painted line is a
  stripe (same floor both sides); a chair/jersey/banner boundary is a step
  (different stuff each side) and is rejected by construction. Says nothing
  about brightness, so white-on-wood, white-on-blue and coloured lines all
  respond the same.

Neither knows what a COURT is — that discrimination belongs to the solve,
which is the only stage that can ask "do these eight lines close into an
18x9m rectangle under one homography".
"""

import cv2
import numpy as np


def detect_segments(frame, y_min=200, limit=30):
    """Return merged white segments [(x1,y1,x2,y2), ...] below y_min.
    limit caps how many (longest first); raise it to audit what's really there."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # white lines = locally bright: tophat picks thin bright structures off wood
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
    _, mask = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
    mask[:y_min] = 0
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=40,
                           minLineLength=40, maxLineGap=8)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4)
    return _merge(segs, limit=limit)


def stripe_response(frame, widths=(3, 5, 8, 12), blur=3):
    """Per-pixel "I am the middle of a stripe" score, in Lab colour distance.

    For each half-width d and each of 4 axes, compare the pixel to its
    neighbours at +-d: a stripe centre differs from BOTH flanks while the
    flanks match each other. Scoring flank agreement is what rejects step
    edges — at a chair edge the two flanks are wildly different, so the score
    goes negative no matter how strong the edge is.

    Several widths because a 5cm line is ~15px near the camera and ~2px at the
    far endline; the old constant-width assumption is exactly what made far
    lines invisible and fat highlights look like lines.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(frame, (blur, blur), 0), cv2.COLOR_BGR2Lab)
    lab = lab.astype(np.float32)
    best = np.full(lab.shape[:2], -1e9, np.float32)
    for d in widths:
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a = np.roll(np.roll(lab, d * dy, 0), d * dx, 1)
            b = np.roll(np.roll(lab, -d * dy, 0), -d * dx, 1)
            centre = np.minimum(np.linalg.norm(lab - a, axis=2),
                                np.linalg.norm(lab - b, axis=2))
            flank = np.linalg.norm(a - b, axis=2)
            np.maximum(best, centre - flank, out=best)
    return best


def floor_mask(frame, k=5, close=25):
    """Mask of the playing surface: the dominant colour region of the lower frame.

    The stripe test alone is necessary but NOT sufficient — net mesh, jersey
    stripes, chair legs and banner text are all genuinely stripes and pass it
    on merit. What separates them from court lines is that none of them are on
    the floor. k-means the Lab colours, keep the cluster owning the most of the
    bottom third, close over players standing on it.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 4, h // 4))
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Seeded: KMEANS_PP_CENTERS draws from OpenCV's global RNG, so this mask —
    # and therefore calib_score, court_search's cost map and every stripe
    # detection — used to change run to run. Measured on out/testimgs/test_008:
    # the SAME pose scored net error 0.95px and 76.58px on consecutive calls,
    # i.e. pass and fail. Any "measured" comparison taken before this seed was
    # partly reading the RNG.
    cv2.setRNGSeed(0)
    _, lbl, _ = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(small.shape[:2])
    bottom = lbl[int(lbl.shape[0] * 0.66):]
    # Keep EVERY substantial colour of the lower frame, not just the dominant
    # one. A coloured court inside a contrasting apron is the norm in broadcast
    # volleyball, and argmax keeps only one of the two. Measured on
    # out/testimgs: a red-court/teal-apron frame masked 11% of the frame where
    # single-tone floors mask 38-57%, which deleted the court's own lines from
    # the evidence map and made calib_score reject a visibly CORRECT pose.
    share = np.bincount(bottom.ravel(), minlength=k) / bottom.size
    keep = np.flatnonzero(share >= 0.20)
    if not len(keep):
        keep = [int(share.argmax())]
    m = np.isin(lbl, keep).astype(np.uint8) * 255
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    # close over players/shadows, then keep only the largest blob
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        m = (labels == big).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close * 2, close * 2), np.uint8))


def detect_stripes(frame, y_min=0, limit=200, thresh=12.0,
                   widths=(3, 5, 8, 12), min_len=40, max_gap=8, on_floor=True):
    """Stripe-based line segments [(x1,y1,x2,y2), ...]. See stripe_response.
    on_floor restricts to the playing surface — see floor_mask for why."""
    resp = stripe_response(frame, widths)
    mask = (resp > thresh).astype(np.uint8) * 255
    if on_floor:
        mask = cv2.bitwise_and(mask, floor_mask(frame))
    mask[:y_min] = 0
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=40,
                           minLineLength=min_len, maxLineGap=max_gap)
    if segs is None:
        return []
    return _merge(segs.reshape(-1, 4), limit=limit)


def grow_objects(frame, tol=9.0, seed_step=4, min_area=40):
    """Region-grow same-colour blobs. Returns (labels int32, [pixel arrays]).

    Flood-fills from a seed grid with a fixed colour tolerance in Lab, so a
    region is "pixels reachable from the seed whose colour stays within tol of
    it". No notion of lines yet — this only asks what the OBJECTS are.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2Lab)
    h, w = lab.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    labels = np.zeros((h, w), np.int32)
    flags = (4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
    pix, nxt = [], 1
    for y in range(0, h, seed_step):
        for x in range(0, w, seed_step):
            if labels[y, x] or mask[y + 1, x + 1]:
                continue
            cv2.floodFill(lab, mask, (x, y), 0,
                          (tol,) * 3, (tol,) * 3, flags)
            x0, y0, ww, hh = cv2.boundingRect(  # region just marked 255
                (mask[1:h + 1, 1:w + 1] == 255).astype(np.uint8))
            if ww == 0:
                continue
            sub = mask[y0 + 1:y0 + hh + 1, x0 + 1:x0 + ww + 1]
            hit = sub == 255
            ys, xs = np.nonzero(hit)
            sub[hit] = 1  # demote so the next fill can't re-enter but won't match 255
            if len(xs) >= min_area:
                labels[y0 + ys, x0 + xs] = nxt
                pix.append(np.stack([x0 + xs, y0 + ys], 1))
                nxt += 1
    return labels, pix


def line_shape(pts):
    """(length, perp_rms, aspect, segment) for a pixel blob, via PCA.

    A straight thin stripe has tiny perpendicular spread relative to its
    length; a curved arc (basketball key) bulges off its own axis and its
    aspect collapses. That is what separates them — no colour involved.
    """
    p = pts.astype(np.float32)
    c = p.mean(0)
    d = p - c
    _, vecs = np.linalg.eigh(np.cov(d.T) + np.eye(2) * 1e-6)
    axis = vecs[:, -1]
    t = d @ axis
    perp = d @ np.array([-axis[1], axis[0]])
    length = float(t.max() - t.min())
    perp_rms = float(np.sqrt((perp ** 2).mean()))
    aspect = length / (2 * perp_rms + 1e-6)
    seg = (*(c + t.min() * axis), *(c + t.max() * axis))
    return length, perp_rms, aspect, seg


def detect_line_objects(frame, tol=9.0, seed_step=4, min_len=45,
                        min_aspect=14.0, max_perp=3.0, limit=200, merge=True):
    """Grow same-colour objects, keep the ones SHAPED like a line.

    Returns (segments, stats) where stats has one row per accepted object
    BEFORE merging. Definition is structural, not photometric: colour only
    decides what belongs to the same object; acceptance is thin + straight.

    merge=True joins collinear fragments, and it matters more than any
    threshold here: a court line's brightness drifts along its length by more
    than the fixed tolerance window allows (measured: every long object's L
    channel spans exactly 2*tol, i.e. growth stops at the colour boundary, not
    at the end of the line), so one line arrives as several pieces. Merging
    took menlo's longest line from 356px to 1072px in a 1280px-wide frame.
    """
    _, blobs = grow_objects(frame, tol, seed_step)
    out = []
    for pts in blobs:
        length, perp_rms, aspect, seg = line_shape(pts)
        if length >= min_len and aspect >= min_aspect and perp_rms <= max_perp:
            out.append((length, seg, len(pts), aspect, perp_rms))
    out.sort(key=lambda r: -r[0])
    out = out[:limit]
    segs = [r[1] for r in out]
    if merge and segs:
        segs = _merge(np.array(segs, np.float32), limit=limit)
    return segs, [(r[0], r[2], r[3], r[4]) for r in out]


def segment_colors(frame, segs, halfwidth=2):
    """Median Lab colour of the band under each segment — the "layer" it
    belongs to. Court lines share one colour; clutter doesn't."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    out = []
    for x1, y1, x2, y2 in segs:
        n = max(int(np.hypot(x2 - x1, y2 - y1)), 2)
        xs = np.clip(np.linspace(x1, x2, n).astype(int), 0, frame.shape[1] - 1)
        ys = np.clip(np.linspace(y1, y2, n).astype(int), 0, frame.shape[0] - 1)
        out.append(np.median(lab[ys, xs], axis=0))
    return np.array(out, np.float32) if out else np.zeros((0, 3), np.float32)


def _merge(segs, angle_tol=np.deg2rad(3), dist_tol=12, limit=30):
    """Greedy merge of nearly-collinear overlapping segments."""
    used = np.zeros(len(segs), bool)
    out = []
    for i in range(len(segs)):
        if used[i]:
            continue
        group = [segs[i]]
        used[i] = True
        a1 = np.arctan2(segs[i][3] - segs[i][1], segs[i][2] - segs[i][0])
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            a2 = np.arctan2(segs[j][3] - segs[j][1], segs[j][2] - segs[j][0])
            d = abs((a1 - a2 + np.pi / 2) % np.pi - np.pi / 2)
            if d > angle_tol:
                continue
            # midpoint distance to line i
            mx, my = (segs[j][0] + segs[j][2]) / 2, (segs[j][1] + segs[j][3]) / 2
            n = np.array([-(segs[i][3] - segs[i][1]), segs[i][2] - segs[i][0]], float)
            n /= np.linalg.norm(n)
            if abs(n @ [mx - segs[i][0], my - segs[i][1]]) < dist_tol:
                group.append(segs[j])
                used[j] = True
        pts = np.array(group).reshape(-1, 2)
        # fit line through all endpoints, take extremes along direction
        vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        t = (pts - [x0, y0]) @ [vx, vy]
        p1 = (x0 + t.min() * vx, y0 + t.min() * vy)
        p2 = (x0 + t.max() * vx, y0 + t.max() * vy)
        out.append((*p1, *p2))
    # longest first, cap for readability
    out.sort(key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
    return out[:limit]


def draw_segments(frame, segs):
    out = frame.copy()
    for k, (x1, y1, x2, y2) in enumerate(segs):
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.putText(out, str(k), (mx + 3, my - 3), 0, 0.7, (0, 255, 0), 2)
    return out


def _self_check():
    """The definition, as a test: a stripe scores high, a step edge does not."""
    floor = np.full((200, 300, 3), (60, 90, 140), np.uint8)  # wood-ish BGR

    stripe = floor.copy()
    stripe[95:101, :] = (240, 240, 240)  # white painted line, floor both sides
    s = stripe_response(stripe)[98, 150]

    step = floor.copy()
    step[100:, :] = (240, 240, 240)  # boundary: floor above, white below
    e = stripe_response(step)[100, 150]

    assert s > 12, f"painted stripe scored {s:.1f}, should pass threshold"
    assert e < 12, f"step edge scored {e:.1f}, should be rejected"

    # and it must not care about brightness: a DARK line on a light floor is
    # still a line (the old tophat definition misses these entirely)
    light = np.full((200, 300, 3), (230, 230, 225), np.uint8)
    light[95:101, :] = (40, 60, 90)
    assert stripe_response(light)[98, 150] > 12, "dark-on-light line missed"

    segs = detect_stripes(stripe, thresh=12.0)
    assert segs, "no segment found on a synthetic painted line"

    # region-grow + shape: the straight stripe is a line, a fat blob and a
    # curved arc of the SAME colour are not
    blob = floor.copy()
    cv2.circle(blob, (150, 100), 40, (240, 240, 240), -1)
    cv2.ellipse(blob, (150, 100), (70, 70), 0, 200, 340, (240, 240, 240), 5)
    got, _ = detect_line_objects(stripe)
    assert got, "region-grow missed the straight stripe"
    assert not detect_line_objects(blob)[0], "accepted a disc/arc as a line"

    # floor_mask must be reproducible: its k-means used to read OpenCV's global
    # RNG, which made every score built on it (calib_score, court_search) vary
    # between identical runs.
    noisy = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert np.array_equal(floor_mask(noisy), floor_mask(noisy)), \
        "floor_mask is not deterministic"

    print(f"lines self-check ok: stripe {s:.0f} > 12 > step {e:.0f}, "
          f"{len(segs)} segment(s) on the synthetic line, floor_mask deterministic")


def intersect(seg_a, seg_b):
    """Intersection point of the infinite lines through two segments."""
    x1, y1, x2, y2 = seg_a
    x3, y3, x4, y4 = seg_b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return (px, py)


if __name__ == "__main__":
    _self_check()

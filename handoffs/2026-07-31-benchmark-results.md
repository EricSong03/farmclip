# Benchmark results: no method calibrates an unseen venue

Written 2026-07-31, completing both jobs of
`2026-07-31-verify-flip-and-benchmark.md`.

## Headline

**0 of 8 held-out frames get a usable calibration, from any of the four
methods.** Not "needs tuning" — none of them are close.

| method | pass | kp_px | net_px | floor_m | cam_m | f_ratio | secs |
|---|---|---|---|---|---|---|---|
| v6b keypoints | 0/11 | 268 | 112 | 14.5 | 24.5 | 1.01 | 0.4 |
| lineseg (init=v6b) | 0/11 | 1576 | 1912 | 13.6 | 26.5 | 3.58 | 1.3 |
| classical (Hough) | 0/11 | — | — | — | — | — | 0.2 |
| pose search | 0/11 | 523 | 377 | 21.1 | 32.3 | 0.54 | 17.3 |

Pass = keypoints within 20px AND floor within 1 m of the ground-truth pose.
Classical produced **no pose at all** on any frame — no `--` there is a
formatting gap, it never got past the net-band gate. lineseg inherits v6b's
failure by construction (it refines, it does not find) and then amplifies it:
median focal 3.7× ground truth. Pose search does gain from the convention flip
but lands in the wrong basin every time, which is what its own ponytail note
already predicted for blind global search.

**The single most useful frame is `test_020`.** v6b detects 12 keypoints and
fits them at **7.1px reprojection** — its best score in the set — and puts the
court on the right-hand half of the real court, **572px** from truth. Open
`out/bench/v6b/test_020.jpg`. That one picture is why this benchmark exists,
and it reproduces the pre-flip finding (16/22 frames with ≥5 keypoints, zero
usable calibrations) rather than contradicting it, which is the sanity check
the previous handoff asked for.

## What was fixed on the way (both were silently corrupting measurements)

**`floor_mask`'s k-means was never seeded.** It draws from OpenCV's global RNG,
so `calib_score`, `court_search`'s cost map and every stripe detection returned
different answers on identical input. Measured on `test_008`: the same pose
scored net error 0.95px and 76.58px on consecutive calls — pass and fail. Every
"measured" comparison in this repo that touched the floor mask was partly
reading the RNG. One `cv2.setRNGSeed(0)` in `lines.floor_mask` fixes it for all
callers; `farmclip.lines`'s self-check now asserts it.

**`floor_mask` kept only ONE floor colour.** A coloured court inside a
contrasting apron — a red court on a teal surround, an orange court on blue —
is the norm in broadcast volleyball, and k-means argmax kept one of the two.
Measured on `out/testimgs`: two-tone frames masked 11–13% of the frame where
single-tone floors mask 38–57%, which deleted the court's own painted lines
from the evidence map. That made `calib_score` reject `test_016` and
`test_021`, whose overlays are **visibly correct**. Now keeps every cluster
holding ≥20% of the lower frame.

This one is worth dwelling on: the metric was wrong in the *safe-looking*
direction. A proxy that says "this pose is bad" when it is good costs you
ground truth silently, and the only reason it was caught is that the rejected
overlay got opened. The gate in `benchmark.py` now VETOES clear failures rather
than requiring clear successes, and its coverage threshold (0.50) is calibrated
against frames confirmed by eye — named in the code so the next person can
recheck them.

**8 of the 19 test labellings are still not ground truth.** `benchmark.py`
gates them and prints why:

- `test_002/003` — every click in ONE vertical plane (one sideline). That does
  not constrain a pose; PnP picks a member of a family and reports a tiny
  residual for it. They fit at **1.4 and 4.2px** and project courts that miss
  the paint entirely. **Fix: click the other sideline.**
- `test_012` — 6 m of depth span, so focal and distance trade off exactly.
  **Fix: click the far end of the court.**
- `test_009` — floor fits, net 73px off. Re-solving under each naming shows
  **the L/R flip puts it on the paint: press `g` in `calib.html`.**
- `test_014` — 8 clicks, full depth and width span, 10.4px residual, and the
  overlay is still wrong on two independent visual reads. No naming flip
  rescues it, so the clicks themselves are misplaced. **Fix: relabel from
  scratch.**
- `test_000/013/015` — not relabelled; currently absent from the set.

The live watcher now catches the first two classes *while you click*
(`width_span`/`depth_span` in `farmclip/court.py`, shared by the watcher and
the benchmark gate), so this should not recur. The single-sideline case is the
one that most needed it: it produces the LOWEST residuals in the set, so the
error number actively reassures you while the pose is unconstrained.

11 good frames is a usable benchmark. ~19 would be a comfortable one.

## What Job 1 actually found

Less than the previous handoff expected. The five run `calib.json` files were
already consistent with the post-flip annotations (byte-identical camera
positions except mikasa, which moved 0.6 m when its annotations were
un-mirrored). Only `out/calib.json` (menlo) was genuinely stale, and menlo's
labels are bad under either convention (741px), so it is quarantined as
`out/calib_preflip_menlo.json` rather than re-solved — the downstream scripts
that read it (`lift_ball`, `players_pass`, `assemble_scene`) now fail loudly
instead of quietly producing mirrored world coordinates. The four loose
`out/calib_*.json` scratch files are deleted. Comment-level mislabels
(`court.LINES`, `calib_score.MODEL` still calling −Z "left") are corrected;
stale names are how this whole class of bug started.

The classical path was extracted from `cli.calibrate` into
`hypothesis.solve_frame` so the CLI and the benchmark score the same code.

## v6b's failure is NOT a naming bug — measured, not assumed

`test_020` (12 keypoints, 7.1px reprojection, court on the wrong half) has the
exact signature of half-court aliasing, so it was worth an hour to rule out a
free fix. It is ruled out.

Every global relabelling makes it worse, not better:

| relabelling | kp_px | floor_m |
|---|---|---|
| identity | **173** | 8.5 |
| L/R mirror | 650 | 15.7 |
| near/far swap | 234 | 19.9 |
| both (180° about Y) | 618 | 15.3 |

The decisive number is the next one. Across 8 frames and **82 detections**,
measured against the ground-truth projection of all 18 court keypoints:

- distance to the point it CLAIMS to be: median **153px**
- distance to the nearest court point of ANY name: median **90px**
- detections landing within 3% of frame width of *any* court point: **22 of 82**

If the detections were real features under wrong names, nearest-any would be
small while own-name stayed large. It is not. Three quarters of the detections
are not on a court point at all, so there is no renaming that recovers them. A
lengthwise "off by one line" shift suggested by the residual pattern
(`attack_far`→`corner_far`, 3 occurrences) was tested directly and also lost:
212px vs 173px.

**Conclusion: this is a generalisation failure, not a labelling one.** v6b does
not transfer off its 6 training venues, and no post-processing fixes that.
Retraining or a different approach is the only path — which makes the held-out
set the thing that decides whether either worked.

## Where to go next

1. **Relabel the 11 rejected frames** (in progress). Everything is judged on 8
   frames until this is done, and 8 cannot separate a real improvement from
   noise. Their old clicks are parked in `out/testimgs/labels_rejected/`;
   `calib.html` requeues an image as soon as its label is out of
   `out/testimgs/labels/`.
2. **Then retrain — but weight the new data by ANGLE, not by venue count.**
   Splitting the test set by solved camera geometry:

   | camera | test frames | v6b median | training frames |
   |---|---|---|---|
   | end-on (behind the end line, ~1.5 m) | 4 | **103px** | 79 (72%) |
   | side-on (level with the net, ~5–6 m) | 7 | **395px** | 31 (28%) |

   A 3.8× gap that lines up with training representation. Venue identity
   explains much less: frames from the SAME VOD — same floor, same lighting,
   same markings — vary by up to 2.3× (`w2qR0cHw4kk`: 163px and 332px), so if
   the venue were the dominant factor those pairs would cluster and they do
   not.

   Caveat worth keeping honest: side-on frames are also further away (17–25 m
   vs 10–14 m) and wider, so angle and scale are confounded here and this test
   set cannot separate them. And *both* categories fail — 103px against a 20px
   bar is not a near miss. So angle is the bigger axis, not the only problem.

   Practical consequence: scraping more venues shot end-on will not fix the
   side-on case, and side-on elevated is the standard pro broadcast camera —
   the configuration that matters most for the target footage. Bias the scrape
   (`scripts/web_frames.py`, `scripts/yt_frames.py`) toward it, rebuild,
   retrain, and score here. Anything that does not move the pass count off 0/N
   did not work, whatever its val loss says.
3. **Keep `court_search` in the picture.** It is untrained, so it cannot have a
   generalisation gap, and it was fighting an inverted template until the flip.
   It still lands in the wrong basin from a blind start (496px), which its own
   ponytail note predicts — but `refine()` pulls an 83px seed to 1.2px in the
   self-check. If a retrained detector gets within ~100px, search may close the
   rest. That is a measurement to make, not a plan to commit to.

`court_search` and the uncommitted `ball3d.py` were left alone per the previous
handoff; `court_search` was benchmarked but not modified.

## Verify

```
python -m uv run python -m farmclip.calib_score
python -m uv run python -m farmclip.lineseg
python -m uv run python -m farmclip.lines
python -m uv run python -m farmclip.court_search
python -m uv run python scripts/benchmark.py --self-check
python -m uv run python scripts/benchmark.py
```

The benchmark's own self-check is the one that matters: it compares a pose
against itself (0.0px) and against its Z-mirror, which projects the symmetric
court to nearly the same pixels. The mirror must fail — 283px / 4.8 m. If that
assertion ever passes the mirror, the harness has stopped measuring the one
thing it was built to catch.

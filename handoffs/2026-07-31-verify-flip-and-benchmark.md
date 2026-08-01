# Handoff: verify the court-convention flip, then benchmark v6b

Written 2026-07-31, immediately after commit `d6120a3`. Two jobs, in order.
Job 2 is the valuable one; job 1 exists so job 2 is not built on sand.

---

## What just changed (read this first, it explains everything else)

`farmclip/court.py` defined `_left` as `z = -HW`, "as seen from +X". For a
camera on the +X side — which is exactly what `_near` asserts — world −Z
projects to the **image right**. So the model's "left" was the viewer's right,
and had been since the repo started.

Every mirror-related bug was a patch on that one sign:

- `consistent_names` silently renamed 51 of 52 web images
- `solve_web` carried a mirror search and per-pair L/R majority voting
- the keypoint model scored **697px** against its labels but **17.9px**
  against their mirror, with 46 of 47 detections preferring the mirror
- `out/mikasa/annotations.json` was stored backwards and nobody noticed,
  because the mirror search silently corrected it on every read
- the clicker appeared to ignore human labels — it genuinely did, because the
  mirror was the only physically possible reading available to it

`_left` is now `+Z` = image-left. Measured at the flip: all 12 sampled human
label sets were geometrically **impossible** before (camera below the floor,
25–156px) and all 12 became physical after (2–12px, camera 2–9 m up). Across
all 120: 118 physical, median 4.47px. 139 labelled overlays re-solve at
**4.75px median**, 126 under 10px.

New: `calibrate.solve_labeled()` — pose from correspondences exactly as given.
No renaming, no mirror enumeration, deterministic. Reports `cam_below_floor`
instead of repairing an impossible labelling. `solve_web`/`solve_auto` keep
their tolerance for naming drift because MODEL detections still have it.

**The deployed v6b ONNX is NOT affected.** Its training labels were already
camera-POV (`canonical_lr` enforced that), so the flip aligned the world model
to the model. Verified: 6/6 v6b detections solve to a physical camera, no
retraining needed.

---

## Job 1 — find what still assumes the old convention (~30 min)

Already checked, do not redo:

- `index.html` — **fine.** Builds the court from `COURT_L`/`COURT_W`
  symmetrically (`±COURT_W/2`) and never uses named left/right points. A
  mirrored world is invisible to a symmetric court.
- `farmclip/ball3d.py` — **fine.** Does not import `court` at all; works in
  image space off `calib["f"]` and `lines.detect_segments`.

### The actual exposure: stale calib.json files

Ten of them, all solved before the flip, all mirrored in Z:

```
out/calib.json  out/calib_auto.json  out/calib_mik_test.json
out/calib_old.json  out/calib_orig.json
out/{dome1,dome2,dome3,dome4,mikasa}/calib.json
```

A mirrored calib still projects the court **correctly** — the court is
symmetric, so the overlay looks right — but every world coordinate derived
from it has Z flipped. That means `scene.json` ball and player positions are
mirrored across the court's long axis while looking perfectly plausible. This
is the exact failure mode that has fooled us all session: correct-looking
picture, wrong numbers.

**Do this:**

1. Re-solve each run from its (now-correct) annotations and overwrite:
   ```
   python -m uv run python -c "
   import json,sys,cv2; sys.path.insert(0,'.')
   from pathlib import Path
   from farmclip.calibrate import solve_labeled
   for r in json.loads(Path('data/runs.json').read_text()):
       ap=Path(r['dir'])/'annotations.json'; ref=Path(r['dir'])/'debug/ref_frame.jpg'
       if not (ap.exists() and ref.exists()): continue
       a={k:v for k,v in json.loads(ap.read_text()).items() if not k.startswith('post_')}
       im=cv2.imread(str(ref)); h,w=im.shape[:2]
       c=solve_labeled(a,w,h)
       print(r['name'], f\"{c['err']:.1f}px\", 'BELOW FLOOR' if c['cam_below_floor'] else f\"cam {c['cam_height']}m\")
   "
   ```
   Expect: mikasa 3.9px, dome1 8.9, dome2 4.5, dome3 8.0, dome4 2.2, all with
   the camera above the floor. **menlo will fail at ~741px — that is known and
   correct**, its labels are bad under either convention and it is already
   excluded by the builder's quality gate. Do not try to fix menlo here.
2. Delete or regenerate the loose `out/calib_*.json` scratch files. They are
   from old experiments and nothing should be reading them; if something is,
   that is the bug.
3. Grep for anything else consuming calibs and reasoning about sides:
   `grep -rn "_left\|_right\|HW" --include=*.py farmclip/ scripts/`
   Anything that hard-codes a Z sign rather than using `court.KEYPOINTS`
   should be looked at.

**Do not touch** `farmclip/court_search.py` or the uncommitted `ball3d.py`
changes — another session owns those.

---

## Job 2 — the benchmark (the important one)

### Why this matters more than it sounds

Every metric this project has invented has, at some point, passed a visibly
wrong pose:

| metric | how it lied |
|---|---|
| reprojection error | a wrong pose scored **1.34px**; a correct one 1.89px |
| line coverage | ranked a correct calib (0.33) **below** a wrong one (0.37) |
| net position | passed a wrong pose at 5.7px that a median frame scored at 262.5px |
| `court_search` objective | scored a court collapsed onto a patch of clutter at **0.55px** |

The common cause: every objective is **one-way** ("is there evidence near the
projected model?"), which is trivially satisfied by shrinking or sliding the
court onto any dense clutter. None of them ask whether the evidence is
*explained* by the model.

The only reason we ever caught these was a human opening a JPEG. That is why
job 2 is worth more than another model.

### The asset that makes it possible

`data/test/` — **19 hand-labelled frames from 11 YouTube VODs, zero overlap
with the 29 training VODs.** Labels in `data/test/labels/*.json`,
provenance in `data/test/sources.txt`. `build_kp_dataset.py` only ever
reads `data/pool`, so this stays held out through any rebuild. `calib.html`
exposes it as a separate pool ("TEST set - held out").

These clicks are the ground truth. `solve_labeled` on them gives a
known-correct pose per frame.

### What to build

`scripts/benchmark.py` — score any method as **pose error against ground
truth**, not against another proxy:

1. Ground truth: `solve_labeled(clicks)` per test frame. Skip any with
   `cam_below_floor` or err > 15px (relabel those instead of scoring against
   them).
2. Candidate: run the method, get a calib.
3. **Score in world metres, not pixels.** Project a fixed grid of court points
   (e.g. `court.KEYPOINTS`) through both poses and report median pixel
   distance, AND back-project the image centre onto the floor plane under both
   to get a metric ground error. Pixel-only scoring hides the focal/height
   ambiguity that has bitten us repeatedly (a pose can fit the floor at 2px
   and put the net 200px wrong).
4. Report per frame and aggregate, and **write overlays** — the number is the
   summary, the picture is the evidence.

### Then measure all four candidates on it

| method | entry point |
|---|---|
| v6b keypoints | `kp_detect.detect_keypoints` → `calibrate.solve_labeled` |
| lineseg | `lineseg.detect_lines` → `lineseg.solve_lines` (needs an init) |
| classical | `lines.detect_line_objects` (stills) |
| pose search | `court_search.search` (untrained, no init needed) |

Expect `court_search` to gain most from the flip — it is a rigid-template
search that was fighting an inverted template — but that is a guess, and the
whole point of this job is to stop guessing.

### Known result to reproduce as a sanity check

Before the flip, v6b on these 22 frames (3 have since been deleted): 16/22
yielded ≥5 keypoints, but **zero produced a usable calibration**, and several
gave 200px+ floor errors at full depth spread. If your harness says v6b is
fine on unseen venues, the harness is wrong — check it before believing it.

---

## Traps that have already cost time

- **Score on a median frame, not a raw one.** `video.median_frame(clip)`
  erases players. Same calib, same metric: 5.7px on a raw frame, 262.5px on
  the median, because players near a wrongly-projected net supply spurious
  evidence. Stills have no motion to median away, so treat still-frame net
  errors as a *lower bound*.
- **Depth spread beats point count.** Clicks spanning < 8 m of court length
  leave focal and camera distance trading off exactly; the solve slides to the
  focal clamp. Measured: points at x=0 and x=+3 only → 78.8px with f pinned;
  one far-end point added → 8.9px with f free.
- **One bad click poisons every point.** Least squares smears it, so the whole
  overlay looks wrong rather than one corner. `label_watch.worst_click` does
  leave-one-out and names it (found a click that took a fit from 340.9px to
  5.8px).
- `python -m uv run ...` — bare `uv` is not on PATH. **ffmpeg is not
  installed**; `yt_frames.py` streams via `yt-dlp -g` + cv2 seek to avoid it.
- Run frame-sampling with `video.read_frames(..., seek=True)` — sequential
  decode is ~34× slower on full-match videos.

## Verify before claiming anything works

```
python -m uv run python -m farmclip.calib_score
python -m uv run python -m farmclip.lineseg
python -m uv run python -m farmclip.lines
python -m uv run python -m farmclip.court_search
```

All four pass as of `d6120a3`. And per `CLAUDE.md`: render overlays into
`out/debug/` and **look at them** before calling any stage done. That rule
exists because it is the only thing that has reliably caught these failures.

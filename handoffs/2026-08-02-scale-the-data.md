# Handoff: the benchmark is trustworthy now, and it says collect data

Written 2026-08-02. The previous handoff asked for a benchmark; this one reports
what it measured once it was working, and what to do about it.

The short version: **nine click-sets stand between here and the next real
answer.** Everything else is built.

---

## Where things stand

- `court-pose-v9` is the current model. On 22 gated held-out frames: **2/22
  usable**, 32.3px on the target PoV, 226.3px on broadcast.
- The dataset is **250 train / 54 val**, up from 216 after one new run landed.
- `data/` holds every label and is fully tracked. `out/` is derived and holds
  nothing that matters.
- Nothing is uncommitted and nothing is unpushed.

## What is waiting on a human

Nine runs need one click-set each, in `calib.html` (run dropdown, or `n`/`r` to
step through):

```
tallones-oos  tallones-vlacup  tallones-vlawest        <- target PoV, local video
bc-lnarena    bc-perugia       bc-wislutheran
bc-pineview   bc-nebraska      bc-ncaaqf               <- elevated broadcast
```

Plus 23 single frames in `data/pool` — worth doing, but worth far less. Do the
runs first and understand why before spending clicks anywhere else.

**Start `label_watch.py` before clicking.** Without it there is no overlay and
no warning about the degenerate labellings, and on a run a bad click is applied
to every frame that run contributes rather than to one image.

---

## Why runs and not more single frames

A run is one annotation covering a whole venue: the camera is fixed, so one
clicked reference frame labels every frame sampled from that clip. `dome1`
contributes 41 training frames for one click-set. `tallones-evl`, the run
labelled while writing this, contributed **41 frames for one click-set** and
took the dataset from 216 to 250 by itself.

That matters because of what the venue sweep measured. Five models were trained
on 10 / 20 / 40 / 60 / 81 source venues with frames per venue capped at 3, so
that venue count varied roughly independently of dataset size:

| venues | frames | target PoV | broadcast |
|---|---|---|---|
| 10 | 9 | 654px | 630px |
| 20 | 20 | 203px | 457px |
| 40 | 36 | 221px | 345px |
| 60 | 66 | 69px | 353px |
| 81 | 92 | 89px | 289px |
| **81 (uncapped = v9)** | **216** | **32px** | **226px** |

The venue slope is unreadable — 60 beats 81, so noise dominates at these sizes
and every capped model is undertrained. That half of the experiment failed.

The last row is the half that worked, and it is an accident: same 81 venues, 92
frames against 216, differing only in how densely each venue is sampled. Error
falls 3x and it is the only configuration that produces a usable pose at all.
**Frames per venue is the lever; venue count is not.** Nine runs is the cheapest
way to add frames per venue that exists.

## The two groups scale completely differently

Fitted separately, both cleanly:

- **target PoV: error ~ N^-0.90** (R2=0.94) -> 20px at roughly **370 frames**
- **broadcast: error ~ N^-0.31** (R2=0.96) -> 20px at roughly **565,000 frames**

Do not read the second number as a target. Read it as: broadcast is not
improving from the data we have been adding, and no realistic amount of it will
change that.

The likely reason is imbalance, not difficulty. The training pool is 46% dead
end-on and 50% floor level, with **one frame** above 25 degrees of elevation,
while every broadcast test frame is shot from up in the stands. So each frame
added lands in-distribution for target and out-of-distribution for broadcast.

The competing explanation — that broadcast is intrinsically harder because the
camera is 17-25m back — predicts the wrong sign. A smaller court in frame means
the same world error produces FEWER pixels, not more. Distance cannot explain
226px.

**The six `bc-*` runs are the test of this.** If broadcast lands near 30-50px
once it has ~200 in-distribution frames, imbalance was the whole story. If it
stalls near 150px, elevated views are genuinely harder and the architecture is
implicated — worth knowing before collecting another thousand frames.

---

## What is already built

- `scripts/benchmark.py` — scores any ONNX against the held-out set as pose
  error in pixels AND metres. `--model` to score a candidate without deploying
  it, `--self-check` before trusting a run.
- `scripts/bench_report.py` — renders `results_*.json` to HTML.
- `scripts/v9_gallery.py` — 20 train frames beside 20 held-out, as pictures.
  `reports/v9-fit.html` is the current one: **6px on train, 58px on held-out.**
- `scripts/check_data.py` — fails if anything under `data/` is ignored or
  untracked. Run it before a training run or a handoff.
- `scripts/add_runs.py` — registers new runs; refuses any VOD backing a test
  frame.
- `build_kp_dataset.py` — `--venues N --seed S` for ablations, `--max-per-venue`
  to make the venue axis interpretable, `--path` for the GPU box's mount.

## Traps that have already cost time

- **A low click residual is not evidence of a correct pose.** Clicks along one
  sideline lie in a single plane and constrain nothing; the three worst frames
  in the original test set fitted at 1.9 / 0.4 / 1.5px, the LOWEST residuals
  recorded, and projected courts that missed the paint entirely. `label_watch`
  warns live now.
- **Check the training pool before reserving a test venue.** `tallones-usav`
  and `tallones-vladimes` were reserved as held-out while 12 of their frames
  sat in training from an earlier session, which silently turned v8's headline
  7/10 into a memorisation score. `build_kp_dataset` blocks this now and prints
  what it excluded.
- **`git status` does not report what git was told to ignore.** Data has gone
  missing four times this way: the web frames, four dome venues' annotations,
  `data/runs` (eaten by an unanchored `runs/` pattern), and the ball CSVs. All
  four were invisible. `check_data.py` is the only thing that catches it.
- **`build_kp_dataset` used to never delete.** It reported 125 web images while
  the directory held 160, so every model trained here was fed images its own
  quality gates had rejected. Fixed; disk now matches the report exactly.
- **Only 2 of 12 videos are tracked.** Ablation splits must be built on the
  machine that has `videos/` and copied to the GPU box, never rebuilt there.
- `python -m uv run ...` — bare `uv` is not on PATH. Piped stdout is buffered,
  so a redirected benchmark run looks frozen for minutes.

## Measured dead ends — do not retry from scratch

- **Keypoints seeded into `court_search.refine`.** 9 of 20 frames degraded, 0
  improved; 57.7px -> 283.3px. The damage lands on the BEST seeds. `guard_img`
  does not save it. The seed was never the constraint, the objective is wrong.
- **Relabelling schemes for v6b's failure.** Every global renaming scores worse
  than identity, and across 82 detections the median distance to the nearest
  court point of ANY name is 90px. They are not real features under wrong
  names.
- **`classical`** produces no pose on any held-out frame. **`lineseg`** inherits
  the keypoint pose and amplifies its error.

---

## The next three steps, in order

1. **Label the nine runs.** Do one `bc-*` run first and rebuild — a league feed
   cuts to replays and other cameras, and the drift gate drops those, so the
   yield per broadcast run is unknown. Measure it before committing to six.
2. **Retrain on the result** with v9's recipe (`epochs=700 patience=250 batch=8
   imgsz=1280`), export ONNX, and score with `--model`. Judge only the held-out
   number; the val split shares venues with train and that gap is the entire
   problem.
3. **Read the two groups separately.** Target PoV near 20px means the scaling
   law held and collecting more works. Broadcast still near 200px after six
   in-distribution runs means the ceiling is the approach, not the dataset —
   and that is the moment to stop collecting and change something else.

## Verify before believing anything

```
python -m uv run python scripts/check_data.py
python -m uv run python scripts/benchmark.py --self-check
python -m uv run python -m farmclip.calib_score
python -m uv run python -m farmclip.lineseg
python -m uv run python -m farmclip.lines
python -m uv run python -m farmclip.court_search
```

The benchmark's self-check is the one that matters: it scores a pose against
itself (0.0px) and against its Z-mirror, which projects the symmetric court to
nearly the same pixels and must fail at 283px / 4.8m. If that ever passes the
mirror, the harness has stopped measuring the one thing it exists to catch.

And per `CLAUDE.md`: render overlays and **look at them**. Three separate metric
bugs this week were caught by opening a JPEG and by nothing else.

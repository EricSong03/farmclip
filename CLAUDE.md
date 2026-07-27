# farmclip

Turn stationary(ish) volleyball gameplay video into a 3D model of the game. Current scope: **court lines + net only**. Players/ball come later.

## Current state

- `index.html` — the whole app so far. No build step, no dependencies to install (three.js from CDN). Open directly in a browser.
  - Left: video upload (local object URL, nothing leaves the machine) + canvas overlay for clicking calibration points.
  - Right: three.js render of a regulation FIVB court (18×9 m, attack lines at 3 m, net at 2.43 m) with orbit controls.
  - "Calibrate" is a stub — logs clicked points to console.
- `label.html` — ball-label checker (Chrome/Edge): pick the repo folder once, it auto-loads clips + label csvs (runs hardcoded in `RUNS`), serves random suspect frames balanced across clips, click to correct / `x` to reject, auto-saves to `out/finetune/ball/<name>.labeled.csv` (resumes from it).
- `examples/` — reference screenshots/clips of real PoVs we're targeting.

## Chosen approach (see docs/specs/ and docs/plans/roadmap.md "Decisions")

Two halves, one seam:

- **Python pipeline** (uv-managed, CPU-only, offline): video in → per-frame scene JSON out. Does auto calibration (existing court keypoint models → solvePnP), ball detection (TrackNet vs YOLO bake-off) + ballistic 3D lifting, player tracking → ground positions (skeletons later).
- **Browser viewer** (static page, three.js): loads scene JSON, timeline scrubber + orbit camera. Being built in a separate session; plan in docs/plans/3d-viewer.md.
- **Seam:** per-frame samples at video fps — `{t, ball:[x,y,z], players:[{id, pos:[x,y,z]}]}` + header with court dims, net height, fps, camera pose.

Calibration solves camera pose from known 3D court-model points — **not** corner-clicking (corners usually out of frame) and **not** classical Hough lines (multi-sport floors are full of distractor lines). Off-floor points (net top, antenna tips) are valuable — use them.

Acceptance per stage: render overlay debug frames (projections drawn on real footage) into `out/debug/` and visually inspect them before calling a stage done.

## Conventions

- Keep it minimal: static front end, offline Python pipeline, no server.
- Court model constants live in `index.html` (`COURT_L`, `COURT_W`, `NET_H`, ...). Meters, Y-up, origin at court center under the net.
- Docs: `docs/specs/` for what to build, `docs/plans/` for how/when, `docs/known-issues.md` for gotchas discovered from real footage.

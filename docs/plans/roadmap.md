# Roadmap

## North star

Full 3D reconstruction of a rally from one stationary camera: accurate court + net, physics-fit ball trajectory, per-player 3D skeletons pinned to the ground plane — with a timeline scrubber to select frames and free orbit/pan camera. Stylized reconstruction (skeletons/capsules), not photoreal.

Build order: **court → ball → players.** Each stage demos on its own and feeds the next (ball lifting needs calibration; player placement needs the calibrated ground plane).

## Decisions (grilled 2026-07-26)

- **Architecture** (revised 2026-07-26, supersedes "no server"): **local web app.** One command starts FastAPI + uvicorn on localhost serving index.html; user uploads a video in the browser → background job runs the pipeline stages (calibrate → ball → players) with per-stage progress polling → viewer loads the finished scene JSON from the server. Processing stays on the user's machine (CPU-only assumed, private, free); same codebase is hostable later if sharing is ever wanted. Stack: Python 3.12 + uv + pyproject.toml, FastAPI (serves static page + upload/progress/scene endpoints, BackgroundTasks — no queue infra), OpenCV, PyTorch/Ultralytics for detectors, NumPy/SciPy for parabola fits. Viewer keeps its local file inputs as a no-server fallback.
- **Input scope:** for now, the first minute of the example vod. Rally auto-detection exists elsewhere and is out of scope here.
- **Calibration:** automatic from day one — try existing open-source volleyball court keypoint/U-Net models first; validate by reprojection (projected lines hug real lines), fine-tune only if they fail. No manual click tool.
- **Ball:** run BOTH TrackNet-family and YOLO (ovml) detectors head-to-head on the same clip; Claude picks the winner by comparison. Lift to 3D via touch-segmented ballistic parabola fits.
- **Players:** v1 = detect + track → feet point → ground-plane position → capsules. v2 = swap in monocular 3D pose (WHAM/4DHumans-class) on the same tracks.
- **Scene format:** per-frame samples at video fps: `{t, ball:[x,y,z], players:[{id, pos:[x,y,z]}]}` + header (court dims, net height, fps, camera pose). Skeletons later add a `joints` array. Viewer session's phase-1 spec adopts this.
- **Acceptance:** autonomous self-verification when run in a loop — each stage renders overlay frames (projected court lines / ball path / player positions drawn on real footage) and Claude inspects them directly (image read) before calling the stage done. Overlay artifacts kept in `out/debug/`.

## In progress

- [ ] 3D viewer: **built, needs a human eyeball in the browser.** All 3 phases of [3d-viewer.md](3d-viewer.md) implemented in index.html: scene-format spec (docs/specs/scene-format.md) + sample fixture (examples/sample-scene.json), JSON loader, ball/capsule rendering with team colors, lerp + gap rule (`GAP_MAX_S=0.15`, gap readout), video-as-master-clock sync, `net_height` applied from JSON. Logic verified headlessly (syntax, sampleTrack lerp/gap/clamp, fixture integrity). Viewer accepts pipeline's `pos:[x,y,z]` as well as `[x,z]`; extra header fields (fps/camera/court) ignored. To verify: open index.html → load examples/sample-scene.json → play.

## Next up

- [x] App skeleton (2026-07-26): `farmclip.py` (uv inline-deps script, no pyproject yet) — `python -m uv run farmclip.py serve` serves index.html + upload/progress/scene endpoints with a stubbed 3-stage pipeline; browser gains "Process on server" button that uploads, polls stage progress, and auto-loads the returned scene. Verified end-to-end via API (upload → calibrate/ball/players stages → scene.json → 404s). Real pipeline stages replace the stub loop in `run_pipeline()`. Note: bare `uv` not on PATH on this machine — use `python -m uv`.
- [ ] Auto calibration: survey + run existing volleyball court keypoint models on the example footage; solvePnP; render court-overlay debug frames; self-verify.
- [ ] Scene JSON writer emitting the agreed per-frame format; hand a sample to the viewer session.

## Later

- [x] Ball bake-off (2026-07-26): **VballNet ONNX wins** — 80% detection @62fps CPU vs YOLO11n sports-ball 16% (locked onto static objects, 394px median disagreement). Weights: vendor/fast-vb-tracking/models/VballNetV1_seq9_grayscale_330. Ballistic lift (farmclip/ball3d.py): 33/45 flight segments fitted, reprojected arcs verified on footage (out/debug/ball3d_reproj.jpg).
- [ ] Players v1: detect/track → ground positions → scene JSON capsules. (in progress: farmclip/players.py — YOLO persons + IoU tracker + floor-ray)
- [ ] Players v2: monocular 3D pose skeletons.
- [ ] Men's/women's net height toggle (2.43/2.24 m).
- [ ] Pan/zoom footage: per-N-frame calibration refresh + pose smoothing.

## Done

- [x] Web research on sports field registration methods (2026-07-26).
- [x] Single-file stub: video upload, corner-click overlay, 3D court + net render (2026-07-26).
- [x] Reviewed example PoV footage; pivoted from corner-clicking to keypoint-based calibration (2026-07-26).
- [x] Roadmap grilled: architecture, scope, calibration, ball, players, scene format, acceptance all decided (2026-07-26).

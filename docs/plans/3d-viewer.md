# Plan: 3D Viewer (timeline scrubber + per-frame scene data)

Goal (from [roadmap](roadmap.md)): a viewer with timeline scrubber + orbit camera that renders per-frame scene data — court static, ball + players from a simple JSON file. Must work today with hand-authored JSON so it doesn't block on calibration/detection.

Decision: **extend `index.html`, no new file, no new dependency.** The right panel already has the court, orbit controls, and a render loop; the viewer is a data format + a scrubber + per-frame object updates layered on top. (Separate `viewer.html` rejected: would duplicate the whole scene block and violate the single-file convention in CLAUDE.md.)

## Decisions (grilled 2026-07-26)

- **Side-by-side is a requirement, not a layout accident:** the video and the 3D view stay on screen together at the same time so the user can eyeball reconstruction accuracy against the footage. Never collapse to a single-panel mode.
- **Time model:** scene `t` = seconds from video start, 1:1 with `video.currentTime`. No offset field, no frame indices.
- **Master clock:** when a video is loaded, the `<video>` element is the clock — its native controls play/pause/scrub, and the render loop just reads `video.currentTime`. `THREE.Clock` drives playback only when JSON is loaded without a video.
- **Scrubber UI:** none when a video is loaded (native controls are the timeline). A range input + play button appears only for JSON-without-video mode.
- **Data shape:** per-frame snapshots — `frames: [{t, ball, players}]`; a player id missing from a frame is hidden that frame.
- **Teams:** explicit `team: "a" | "b"` on each player; viewer colors by team (never infer from court side — players cross the center line).
- **Gap handling:** between bracketing frames where an object goes missing/null, interpolate if the gap is ≤ `GAP_MAX_S = 0.15` (a few frames of detector flicker); longer gaps hide the object and list the range in a status readout (e.g. "ball: no data 3.2–5.1s"). `GAP_MAX_S` is one named const — a calibration knob to retune from real footage.

## Phase 0 — Findings (done, 2026-07-26)

Source: `index.html` read in full (178 lines).

Allowed APIs / existing patterns to copy, not reinvent:
- three.js **r0.160.0** ESM via import map (`index.html:37-42`); `OrbitControls` from `three/addons/controls/OrbitControls.js` (`index.html:45`). Do NOT change versions or add scripts.
- Scene/camera/renderer/controls/light: `index.html:110-120`. Render loop is `renderer.setAnimationLoop(...)` at `index.html:175` — hook per-frame viewer updates there.
- Court constants `COURT_L, COURT_W, NET_H, NET_W, ATTACK` all on `index.html:108`; court geometry block `index.html:107-164`. Coordinate convention (confirm before math): **X = court length (net at x=0), Z = width, Y = up, meters.**
- UI: controls row `index.html:22-26`, status div `index.html:29`, resize via `ResizeObserver` (`index.html:172`). New controls go in the existing row; new readouts mirror `#status`.
- File loading pattern: `<input type="file">` + `onchange` (`index.html:54-60`). For JSON use `file.text()` + `JSON.parse` — no FileReader ceremony.

Anti-pattern guards (all phases):
- No `THREE.Geometry` (removed in modern three); use `BufferGeometry`-backed primitives (`SphereGeometry`, `CylinderGeometry`) like the existing code.
- No npm/bundler/package.json; no additional CDN libraries.
- Don't add a third `ResizeObserver` — two exist (`index.html:105`, `index.html:172`).
- Don't invent scene-JSON fields not in the spec; version the format instead.

## Phase 1 — Scene format spec + sample data

**Implement:**
1. `docs/specs/scene-format.md` defining v1 JSON:
   - Top level: `{ "version": 1, "net_height": 2.43, "frames": [...] }`
   - Frame: `{ "t": <seconds from video start>, "ball": [x, y, z] | null, "players": [{ "id": "a1", "team": "a", "pos": [x, z] }] }`
   - Units meters; axes exactly as `index.html:108` block (X length, Y up, Z width, origin court center under net). Frames sorted by `t`. Everything except `version`/`frames` optional. Spec documents the gap-handling contract (`GAP_MAX_S`) and that joints/skeletons are a future `version: 2` — noted, not defined.
2. `examples/sample-scene.json` — hand-authored ~3 s rally stub: ball on a serve-like parabolic arc crossing the net (peak above `NET_H`), 2 players per side standing at plausible court positions. This is the test fixture for Phase 2.

**Verification:** JSON parses (`python -c "import json;json.load(open('examples/sample-scene.json'))"` or browser console); all coords within court bounds ±ATTACK sanity; ball `y` peaks > 2.43.

## Phase 2 — Viewer core in index.html

**Implement (all in `index.html`):**
1. Controls row (`index.html:22-26`): add `<input type="file" id="sceneFile" accept=".json">`. A play/pause button + `<input type="range">` scrubber exist too but are shown **only when no video is loaded** (video's native controls are the timeline otherwise).
2. Load: `sceneFile.onchange` → `JSON.parse(await f.text())`; reject `version !== 1` with a `#status`-style message; set scrubber max to last frame `t`.
3. Dynamic objects, created once on load, updated per frame:
   - Ball: `SphereGeometry(0.105)` (regulation radius), yellow `MeshLambertMaterial`; hidden when no valid data (null or large gap).
   - Players: one `CylinderGeometry(0.25, 0.25, 1.8)` per player id, positioned at `(x, 0.9, z)`, colored by `team` (two fixed colors). Capsules/skeletons later.
4. Playback: in the existing `setAnimationLoop` callback (`index.html:175`), get current time `t` — from `video.currentTime` if a video is loaded, else from a `THREE.Clock`-driven `playT` (scrubber sets it directly). Find bracketing frames, **lerp** matching ball/player positions. Gap rule: object present in both bracketing frames → lerp; missing on either side → if gap span ≤ `GAP_MAX_S = 0.15` interpolate across it, else hide and report the range in the status readout. Binary search not needed — linear scan with a cached index cursor is fine at these sizes (`// ponytail:` comment it).

**Docs refs:** copy object-creation style from net/posts block (`index.html:148-164`); loop hook `index.html:175`; spec from Phase 1.

**Verification:** open `index.html` in browser → load `examples/sample-scene.json` (no video): press play, ball flies the arc over the net, players stand court-side colored by team, scrub jumps correctly. Then load the example video too: native video controls now drive the 3D scene, custom scrubber hidden. Bad JSON shows an error, not a blank scene. No console errors.

## Phase 3 — Integration polish

**Implement:**
1. Apply `net_height` from loaded JSON to the net/band/post meshes (`index.html:148-164`) — this also delivers the roadmap's "men's/women's toggle" for free when data supplies 2.24.
2. Gap status readout: a `#status`-style line listing hidden-object ranges (e.g. "ball: no data 3.2–5.1s"), computed once at load time.

**Verification:** JSON with `"net_height": 2.24` visibly lowers the net; a sample with a >0.15s ball gap hides the ball during it and lists the range.

## Final phase — Verify

- `grep -n "THREE.Geometry\|<script src" index.html` → no matches (import-map module script only).
- Re-read `docs/specs/scene-format.md` vs. the loader code: every field read by the code exists in the spec, and vice versa.
- Full manual pass of Phase 2 + 3 verification steps in one sitting.
- Update `docs/plans/roadmap.md`: check off the viewer item, add follow-ups discovered.

## Out of scope (deliberately)

- Skeleton rendering, ball trails, camera-pose visualization of the calibrated camera — all after calibration lands.
- Export of scene JSON from the calibration side — that's the calibration workstream's job; the format spec is the contract between them.

# Spec: Scene JSON format (v1)

The contract between the offline pipeline (writer) and the browser viewer (reader). One file per processed clip.

## Coordinates

Meters, Y-up, origin at court center under the net. **X = court length** (net plane at x=0), **Z = court width**. Matches the constants block in `index.html` (`COURT_L=18`, `COURT_W=9`, `NET_H`).

## Time

`t` = seconds from the start of the video file, 1:1 with `video.currentTime`. No offsets, no frame indices. Frames sorted ascending by `t`, sampled at (roughly) video fps — the viewer interpolates, so exact spacing doesn't matter.

## Shape

```json
{
  "version": 1,
  "net_height": 2.43,
  "frames": [
    {
      "t": 0.033,
      "ball": [x, y, z],
      "players": [
        { "id": "a1", "team": "a", "pos": [x, z] }
      ]
    }
  ]
}
```

- `version` (required): `1`. Readers reject anything else.
- `net_height` (optional, default 2.43): 2.24 for women's.
- `frames` (required): may be empty (court-only scene).
- `ball` (optional per frame): `[x, y, z]` or `null`/absent when not detected.
- `players` (optional per frame): visible players only — an `id` absent from a frame is hidden that frame. `pos` is the ground-plane position of the feet: `[x, z]`, or `[x, y, z]` with y ignored by the capsule renderer (pipeline emits 3D; both accepted). `team` is `"a"` or `"b"`, assigned by the pipeline, stable across frames per `id`.
- Additional top-level header fields (`fps`, `court`, `camera` pose) are allowed and ignored by the viewer today; the pipeline may emit them for overlay/debug tooling.

## Gap contract (viewer behavior, writers take note)

Between two frames that both contain an object, the viewer lerps. If an object disappears and reappears within `GAP_MAX_S = 0.15` s, the viewer interpolates across the gap (detector flicker). Longer gaps: object hidden, range reported in the UI. Writers should therefore emit honest nulls/omissions, not held last-known positions.

## Future (v2, noted not defined)

- `joints`: per-player 3D skeleton array (replaces capsule rendering).
- Camera pose + court-dims header for overlay rendering.

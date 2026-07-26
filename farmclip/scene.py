"""Scene JSON writer. Conforms to docs/specs/scene-format.md (v1)."""

import json
from pathlib import Path


def write_scene(
    dst: str | Path,
    frames: list[dict],
    net_height: float = 2.43,
) -> Path:
    """frames: [{"t": s, "ball": [x,y,z] | None, "players": [{"id","team","pos":[x,z]}]}].

    Omits null ball / empty players per spec ("honest nulls/omissions").
    """
    out_frames = []
    for f in sorted(frames, key=lambda f: f["t"]):
        entry = {"t": round(f["t"], 4)}
        if f.get("ball") is not None:
            entry["ball"] = [round(v, 3) for v in f["ball"]]
        if f.get("players"):
            entry["players"] = [
                {"id": p["id"], "team": p["team"], "pos": [round(v, 3) for v in p["pos"]]}
                for p in f["players"]
            ]
        out_frames.append(entry)

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        json.dumps({"version": 1, "net_height": net_height, "frames": out_frames}),
        encoding="utf-8",
    )
    return dst

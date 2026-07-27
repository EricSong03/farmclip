"""Export ball training labels for the GPU fine-tune session (CPU, local).

Builds out/finetune/ball/<name>.csv (Frame,Visibility,X,Y) from what the
pipeline already trusts: frames where both VballNet ensemble members agree,
plus single-model detections backed by an accepted ballistic 3D fit.

Usage: python -m uv run python scripts/export_finetune_data.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
AGREE_PX = 12.0

RUNS = [  # (out dir, export name)
    (ROOT / "out", "menlo"),
    (ROOT / "out" / "mikasa", "mikasa"),
]


def physics_frames(outdir):
    """Frame indices with an accepted ballistic 3D ball fit."""
    b3d = outdir / "ball3d.json"
    if b3d.exists():
        return {int(k) for k in json.loads(b3d.read_text())}
    scene = json.loads((outdir / "scene.json").read_text())
    return {i for i, fr in enumerate(scene["frames"]) if fr.get("ball")}


def export(outdir, name, dest):
    dfs = []
    for sub in ("ball0", "ball1"):
        csvs = list((outdir / sub).glob("*/ball.csv"))
        if not csvs:
            print(f"{name}: missing {sub}, skipping run")
            return None
        dfs.append(pd.read_csv(csvs[0]))
    a, b = dfs[0], dfs[1]
    n = min(len(a), len(b))
    a, b = a.iloc[:n].copy(), b.iloc[:n]
    fit = physics_frames(outdir)

    both = (a.Visibility > 0) & (b.Visibility > 0).values
    close = ((a.X - b.X.values) ** 2 + (a.Y - b.Y.values) ** 2) ** 0.5 <= AGREE_PX
    in_fit = a.Frame.isin(fit)
    only_a = (a.Visibility > 0) & ~(b.Visibility > 0).values & in_fit
    only_b = ~(a.Visibility > 0) & (b.Visibility > 0).values & in_fit
    keep = (both & close) | only_a | only_b

    out = a[["Frame", "X", "Y"]].copy()
    out.loc[only_b, ["X", "Y"]] = b.loc[only_b, ["X", "Y"]].values
    out["Visibility"] = keep.astype(int)
    out.loc[~keep, ["X", "Y"]] = 0
    out = out[["Frame", "Visibility", "X", "Y"]].astype(int)
    out.to_csv(dest / f"{name}.csv", index=False)
    print(f"{name}: {keep.sum()}/{n} labeled "
          f"({int((both & close).sum())} consensus, {int((only_a | only_b).sum())} fit-backed)")
    return keep.sum()


def main():
    dest = ROOT / "out" / "finetune" / "ball"
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for outdir, name in RUNS:
        got = export(outdir, name, dest)
        total += got or 0
    assert total > 1000, f"only {total} labels exported — check pipeline outputs"
    print(f"bundle ready: {dest}")


if __name__ == "__main__":
    main()

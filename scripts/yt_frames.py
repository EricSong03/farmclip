"""Harvest wide court-view frames from a YouTube VOD, via visual selection.

Two phases, because there is no reliable automatic test for "this frame shows
the court". A scoring heuristic (uniform floor area + long straight lines on
it) was tried and measurably failed: on a World Championship VOD its TOP-ranked
frame was a coach in front of a yellow LED advertising board — uniform region,
many parallel lines — while a real court view ranked below it. Broadcast VODs
are full of graphics overlays, LED boards and banners that satisfy any
low-level "flat surface with lines" test. So a human (or an agent that can see)
picks from contact sheets instead.

  stage:  python -m uv run python scripts/yt_frames.py stage <url> [--n 48]
          -> out/ytstage/<id>/f<frame>.jpg + sheet_NN.jpg contact sheets
  pick:   python -m uv run python scripts/yt_frames.py pick <id> f1234 f5678 ...
          -> copies those into out/webimgs as web_NNN.jpg + sources.txt lines

Streams rather than downloads: yt-dlp -g gives a URL cv2 can seek, and a
video-only mp4 format needs no ffmpeg merge.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from web_frames import WEB, next_index

ROOT = Path(__file__).parent.parent
STAGE = ROOT / "out/ytstage"
FMT = "bv[height<=1080][ext=mp4]/bv[ext=mp4]/best[ext=mp4]"
COLS, TW, TH, PER_SHEET = 6, 400, 225, 24


def _ytdlp_cmd():
    """yt-dlp if on PATH, else via uv. NOT sys.executable — that is the venv
    python, which has no uv module (uv lives on the system python here)."""
    exe = shutil.which("yt-dlp")
    return [exe] if exe else ["python", "-m", "uv", "run", "yt-dlp"]


def stream_url(url: str) -> str:
    out = subprocess.run(_ytdlp_cmd() + ["-f", FMT, "-g", "--no-warnings", url],
                         capture_output=True, text=True, timeout=300)
    lines = [l for l in out.stdout.splitlines() if l.startswith("http")]
    if not lines:
        raise RuntimeError(f"yt-dlp gave no stream url: {out.stderr[-300:]}")
    return lines[-1]


def video_id(url: str) -> str:
    for key in ("v=", "youtu.be/", "/shorts/"):
        if key in url:
            return url.split(key)[1][:11]
    return url[-11:]


def build_sheets(vid: str):
    """(Re)build contact sheets from whatever frames are staged for vid.

    Separate from stage() so an interrupted or killed staging run is still
    usable — sheets used to be written only after the sampling loop, so a run
    that died partway left frames on disk that nothing could review or pick.
    """
    d = STAGE / vid
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    fps = meta.get("fps") or 25.0
    frames = sorted(int(p.stem[1:]) for p in d.glob("f*.jpg"))
    if not frames:
        print(f"{vid}: no staged frames")
        return 0
    for k in range(0, len(frames), PER_SHEET):
        chunk = frames[k:k + PER_SHEET]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = np.full((rows * (TH + 20), COLS * TW, 3), 30, np.uint8)
        for j, f in enumerate(chunk):
            im = cv2.imread(str(d / f"f{f}.jpg"))
            if im is None:
                continue
            r, c = divmod(j, COLS)
            y = r * (TH + 20)
            sheet[y:y + TH, c * TW:(c + 1) * TW] = cv2.resize(im, (TW, TH))
            cv2.putText(sheet, f"f{f} {f/fps/60:.0f}m", (c * TW + 4, y + TH + 15),
                        0, 0.45, (0, 255, 255), 1)
        p = d / f"sheet_{k // PER_SHEET:02d}.jpg"
        cv2.imwrite(str(p), sheet)
        print(f"sheet -> {p}")
    return len(frames)


def ensure_meta(vid: str, url: str | None = None):
    """Write meta.json for an interrupted stage run (needs url + fps to pick)."""
    d = STAGE / vid
    if (d / "meta.json").exists():
        return
    url = url or f"https://www.youtube.com/watch?v={vid}"
    fps = 25.0
    try:  # metadata only, no stream
        out = subprocess.run(_ytdlp_cmd() + ["--print", "%(fps)s", "--no-warnings", url],
                             capture_output=True, text=True, timeout=180)
        fps = float(out.stdout.strip().splitlines()[-1])
    except Exception as e:
        print(f"  {vid}: fps lookup failed ({e}) — assuming 25")
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"url": url, "fps": fps}, indent=1))
    print(f"  {vid}: meta.json written (fps {fps})")


def stage(url: str, n: int):
    vid = video_id(url)
    d = STAGE / vid
    d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(stream_url(url))
    if not cap.isOpened():
        sys.exit("cv2 could not open the stream")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        sys.exit("no frame count on the stream")
    # written BEFORE sampling: a killed run must still leave pickable frames
    (d / "meta.json").write_text(json.dumps({"url": url, "fps": fps}, indent=1))
    lo, hi = int(total * 0.08), int(total * 0.95)   # skip intro/outro

    saved = []
    for i in range(n):
        f = lo + (hi - lo) * i // max(n - 1, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imwrite(str(d / f"f{f}.jpg"), frame)
        saved.append(f)
        print(f"  staged f{f} @{f/fps/60:.1f}min", flush=True)
    cap.release()
    build_sheets(vid)
    print(f"\n{len(saved)} frames staged in {d}\n"
          f"review the sheets, then: yt_frames.py pick {vid} f<N> f<N> ...")


def pick(vid: str, frames: list[str], dest: Path | None = None, prefix="web"):
    """Promote staged frames. dest defaults to the TRAINING pool (out/webimgs).

    Pass a different dest for held-out test frames — anything landing in
    out/webimgs gets labelled and trained on, which would stop it being a test
    set at all.
    """
    out = Path(dest) if dest else WEB
    out.mkdir(parents=True, exist_ok=True)
    d = STAGE / vid
    meta = json.loads((d / "meta.json").read_text())
    nxt = next_index() if out == WEB else (
        max((int(p.stem.split("_")[-1]) for p in out.glob(f"{prefix}_*.jpg")),
            default=-1) + 1)
    lines = []
    for fr in frames:
        src = d / (fr if fr.endswith(".jpg") else fr + ".jpg")
        if not src.exists():
            print(f"  {fr}: not staged - skipping")
            continue
        name = f"{prefix}_{nxt:03d}.jpg"
        shutil.copyfile(src, out / name)
        t = int(int(fr.lstrip("f").removesuffix(".jpg")) / meta["fps"])
        lines.append(f"{name}\t{meta['url']}&t={t}s")
        print(f"  {fr} -> {name}  (@{t//60}m{t%60:02d}s)")
        nxt += 1
    if lines:
        with (out / "sources.txt").open("a", encoding="utf-8") as fh:
            fh.write("".join(l + "\n" for l in lines))
    print(f"promoted {len(lines)} frames into {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("url")
    s.add_argument("--n", type=int, default=48)
    p = sub.add_parser("pick")
    p.add_argument("vid")
    p.add_argument("frames", nargs="+")
    p.add_argument("--dest", help="output dir (default out/webimgs = TRAINING pool)")
    p.add_argument("--prefix", default="web")
    sh = sub.add_parser("sheets", help="rebuild sheets for staged (or all) vids")
    sh.add_argument("vid", nargs="?")
    a = ap.parse_args()
    if a.cmd == "stage":
        stage(a.url, a.n)
    elif a.cmd == "sheets":
        vids = [a.vid] if a.vid else sorted(p.name for p in STAGE.iterdir() if p.is_dir())
        for v in vids:
            ensure_meta(v)
            n = build_sheets(v)
            print(f"{v}: {n} frames\n")
    else:
        pick(a.vid, a.frames, a.dest, a.prefix)


if __name__ == "__main__":
    main()

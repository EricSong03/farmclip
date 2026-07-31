"""Video IO helpers (OpenCV; no ffmpeg binary needed)."""

from pathlib import Path

import cv2


def clip_video(src: str | Path, dst: str | Path, start_s: float, end_s: float) -> Path:
    """Copy [start_s, end_s) of src to dst (H.264 — browsers can't decode mp4v). Returns dst."""
    src, dst = Path(src), Path(dst)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not out.isOpened():  # no H.264 encoder on this box; mp4v plays in cv2 but not browsers
        out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    n_frames = int((end_s - start_s) * fps)
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    cap.release()
    out.release()
    return dst


def read_frames(path: str | Path, step: int = 1, seek: bool = False):
    """Yield (frame_index, t_seconds, bgr_frame). step>1 skips frames.

    seek=True jumps between samples instead of decoding and discarding the
    frames in between — the difference between seconds and minutes when
    sampling ~10 frames out of a full-match video. Seeks land on the nearest
    keyframe, so the yielded index is where the decoder actually stopped, not
    i*step; only use it when any frame near the target will do (calibration),
    not when frame indices must line up with another stream (ball tracking).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if seek and step > 1:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for target in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok:
                break
            i = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            yield i, i / fps, frame
        cap.release()
        return
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, i / fps, frame
        i += 1
    cap.release()


def median_frame(path: str | Path, n: int = 40, start: int | None = None,
                 span_s: float = 12.0):
    """Per-pixel median over n frames from ONE window — a court with nobody on it.

    The camera is stationary (the premise of this repo), so players, refs and
    the ball are the only things that move and the median erases all of them.
    Every court detector then sees an unoccluded floor instead of one with
    bodies parked on the lines — cheaper and strictly better than averaging
    detections afterwards.

    Windowed, NOT spread over the whole file, and that is not a detail: source
    VODs are frequently multi-shot broadcasts, and a median across camera
    angles is mush. Measured — it destroyed the court in 8 of 13 benchmark
    clips before the window was added. span_s must stay inside one shot; ~12 s
    is long enough for players to move off any given pixel.
    """
    import numpy as np

    info = video_info(path)
    fps = info["fps"] or 30.0
    span = min(int(span_s * fps), info["frames"])
    if start is None:
        start = max((info["frames"] - span) // 2, 0)
    step = max(1, span // n)
    frames = []
    cap = cv2.VideoCapture(str(path))
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(start + k * step, info["frames"] - 1))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return np.median(np.stack(frames), axis=0).astype("uint8")


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info

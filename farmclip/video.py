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


def read_frames(path: str | Path, step: int = 1):
    """Yield (frame_index, t_seconds, bgr_frame). step>1 skips frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, i / fps, frame
        i += 1
    cap.release()


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

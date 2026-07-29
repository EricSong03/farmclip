"""AI keypoint detection: fine-tuned YOLO pose ONNX -> named court points.

Ultralytics pose export convention: output (1, 5+3K, N) — box xywh(4) +
box conf(1) + K keypoints (x, y, conf) per candidate, coords in the
letterboxed input space. Letterbox math copied from ultralytics LetterBox.
"""

import json
from pathlib import Path

import cv2
import numpy as np

_SESSIONS = {}


def _session(model_path: str):
    if model_path not in _SESSIONS:
        import onnxruntime as ort
        _SESSIONS[model_path] = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
    return _SESSIONS[model_path]


def letterbox(frame, size: int):
    """Resize + center-pad to size x size (ultralytics LetterBox, 114 gray).
    Returns padded image, ratio, (pad_left, pad_top)."""
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def parse_pose_output(raw, ratio, pad, conf_min, names) -> dict:
    """raw (1, 5+3K, N) -> {name: (u, v)} in original pixels.
    Highest-box-conf candidate wins; keypoints with conf >= conf_min only."""
    pred = np.asarray(raw)[0].T  # (N, 5+3K)
    j = int(np.argmax(pred[:, 4]))
    if pred[j, 4] < 0.25:  # ponytail: fixed floor; below this there is no court
        return {}
    kpts = pred[j, 5:5 + 3 * len(names)].reshape(-1, 3)
    return {name: (float((x - pad[0]) / ratio), float((y - pad[1]) / ratio))
            for name, (x, y, c) in zip(names, kpts) if c >= conf_min}


def detect_keypoints(frame, model_path="finetune_out/yolo11s-court.onnx",
                     conf_min=0.5,
                     names_path="out/finetune/court/kpt_names.json") -> dict:
    """BGR frame -> {keypoint name: (u, v)} in original pixel coords."""
    sess = _session(str(model_path))
    inp = sess.get_inputs()[0]
    size = inp.shape[2]  # [1, 3, S, S]
    img, ratio, pad = letterbox(frame, size)
    x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    raw = sess.run(None, {inp.name: np.ascontiguousarray(x)})[0]
    names = json.loads(Path(names_path).read_text())
    return parse_pose_output(raw, ratio, pad, conf_min, names)

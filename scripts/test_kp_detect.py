"""Unit test for kp_detect.parse_pose_output — no model, plain asserts.
Usage: python -m uv run python scripts/test_kp_detect.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from farmclip.kp_detect import parse_pose_output
from farmclip.court import KEYPOINTS

names = list(KEYPOINTS)
K = len(names)
assert K == 18
N = 7
ratio, pad = 0.5, (20.0, 10.0)

raw = np.zeros((1, 5 + 3 * K, N), np.float32)
# detection 3 (winner, box conf 0.9):
#   kpt0 at letterboxed (120, 60) conf 0.9 -> orig (200, 100)
#   kpt1 conf 0.2 -> dropped
#   kpt2 at (20, 10) conf 0.5 -> orig (0, 0), boundary conf kept
raw[0, 4, 3] = 0.9
raw[0, 5:8, 3] = (120, 60, 0.9)
raw[0, 8:11, 3] = (300, 300, 0.2)
raw[0, 11:14, 3] = (20, 10, 0.5)
# detection 5: lower box conf but all kpts confident — must be ignored entirely
raw[0, 4, 5] = 0.6
raw[0, 5:5 + 3 * K, 5] = np.tile((500, 500, 0.99), K)

out = parse_pose_output(raw, ratio, pad, conf_min=0.5, names=names)
assert set(out) == {names[0], names[2]}, out
u, v = out[names[0]]
assert abs(u - 200) < 1e-4 and abs(v - 100) < 1e-4, (u, v)
u2, v2 = out[names[2]]
assert abs(u2) < 1e-4 and abs(v2) < 1e-4, (u2, v2)
assert all(isinstance(c, float) for c in (u, v))

# every candidate below the box-conf floor -> empty dict
raw2 = np.zeros((1, 5 + 3 * K, N), np.float32)
raw2[0, 4, :] = 0.1
assert parse_pose_output(raw2, ratio, pad, 0.5, names) == {}

print("test_kp_detect: all asserts passed")

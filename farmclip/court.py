"""FIVB court model. Meters, Y-up, origin at court center under the net.

Matches the viewer's convention (index.html): X along court length, Z across
width, Y up. Camera side is +X (near end line at x=+9).
"""

import numpy as np

COURT_L = 18.0
COURT_W = 9.0
ATTACK = 3.0
NET_H = 2.43  # men's; 2.24 women's
NET_W = 9.5
ANTENNA_H = 0.8  # antenna extends this far above net top
LINE_W = 0.05

HL, HW = COURT_L / 2, COURT_W / 2

# Named 3D keypoints (x, y, z). Left/right as seen from +X (camera side).
# z=-HW is left sideline from camera side, z=+HW right.
KEYPOINTS = {
    "corner_far_left": (-HL, 0, -HW),
    "corner_far_right": (-HL, 0, +HW),
    "corner_near_left": (+HL, 0, -HW),
    "corner_near_right": (+HL, 0, +HW),
    "attack_far_left": (-ATTACK, 0, -HW),
    "attack_far_right": (-ATTACK, 0, +HW),
    "attack_near_left": (+ATTACK, 0, -HW),
    "attack_near_right": (+ATTACK, 0, +HW),
    "center_left": (0, 0, -HW),
    "center_right": (0, 0, +HW),
    "net_top_left": (0, NET_H, -HW),
    "net_top_right": (0, NET_H, +HW),
    "antenna_tip_left": (0, NET_H + ANTENNA_H, -HW),
    "antenna_tip_right": (0, NET_H + ANTENNA_H, +HW),
    "post_top_left": (0, NET_H + 0.12, -NET_W / 2),
    "post_top_right": (0, NET_H + 0.12, +NET_W / 2),
    "post_base_left": (0, 0, -NET_W / 2),
    "post_base_right": (0, 0, +NET_W / 2),
}

# Court line segments as 3D point pairs, for overlay rendering.
LINES = [
    ((-HL, 0, -HW), (+HL, 0, -HW)),  # left sideline
    ((-HL, 0, +HW), (+HL, 0, +HW)),  # right sideline
    ((-HL, 0, -HW), (-HL, 0, +HW)),  # far end line
    ((+HL, 0, -HW), (+HL, 0, +HW)),  # near end line
    ((0, 0, -HW), (0, 0, +HW)),      # center line
    ((-ATTACK, 0, -HW), (-ATTACK, 0, +HW)),  # far attack line
    ((+ATTACK, 0, -HW), (+ATTACK, 0, +HW)),  # near attack line
    ((0, NET_H, -HW), (0, NET_H, +HW)),      # net top band
    ((0, 0, -NET_W / 2), (0, NET_H + 0.12, -NET_W / 2)),  # left post
    ((0, 0, +NET_W / 2), (0, NET_H + 0.12, +NET_W / 2)),  # right post
]


def keypoint_array(names):
    """3D coords for the given keypoint names as float32 (N,3)."""
    return np.array([KEYPOINTS[n] for n in names], dtype=np.float32)

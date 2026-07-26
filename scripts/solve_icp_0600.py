import json

import cv2
import numpy as np

from farmclip.court import HL, HW, ATTACK, NET_H, KEYPOINTS
from farmclip.calibrate import solve, draw_overlay
from farmclip.line_icp import refine
from farmclip.lines import intersect

segs = json.load(open("out/segments_0600.json"))
attack_near = segs[4]
sideline_left = segs[14]
pts = np.vstack([np.array(segs[18]).reshape(2, 2), np.array(segs[15]).reshape(2, 2)])
sideline_right = (*pts[0], *pts[-1])
net_band = segs[2]

anl = intersect(attack_near, sideline_left)
anr = intersect(attack_near, sideline_right)

# init from point solve: floor intersections + measured net-band samples at
# the antenna x-positions read from crops (286 -> left sideline crossing)
init_ann = {
    "attack_near_left": anl,
    "attack_near_right": anr,
    "net_top_left": (286, 274),
    "net_top_right": (947, 260),
    "post_base_left": (189, 396),
}
calib = solve(init_ann, 1280, 720)
print("init err", round(calib["err"], 1), "f", round(calib["f"]))

# line correspondences: 2D segment -> 3D model line
# NOTE: near court faces camera => near attack line at x=+ATTACK; sidelines
# span the near half visible portion (+ side of court), net band z=-HW..HW
line_pairs = [
    (attack_near, ((ATTACK, 0, -HW), (ATTACK, 0, +HW))),
    (sideline_left, ((0, 0, -HW), (HL, 0, -HW))),
    (sideline_right, ((0, 0, +HW), (HL, 0, +HW))),
    (net_band, ((0, NET_H, -HW), (0, NET_H, +HW))),
]
# floor intersections weighted 5x (repeat), plus measured left post base
point_pairs = 5 * [(np.array(anl), np.array(KEYPOINTS["attack_near_left"])),
                   (np.array(anr), np.array(KEYPOINTS["attack_near_right"]))]
point_pairs += [(np.array([189, 396]), np.array(KEYPOINTS["post_base_left"]))]
calib = refine(calib, line_pairs, point_pairs, f_range=(0.5, 1.5))
print("icp err", round(calib["err"], 1), "f", round(calib["f"]))
json.dump(calib, open("out/calib.json", "w"), indent=1)

frame = cv2.imread("out/debug/frame_0600.jpg")
cv2.imwrite("out/debug/calib_icp_overlay.jpg", draw_overlay(frame, calib, init_ann))
print("overlay written")

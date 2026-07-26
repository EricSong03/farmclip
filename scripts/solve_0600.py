import json

import numpy as np

from farmclip.lines import intersect

segs = json.load(open("out/segments_0600.json"))

# Claude's semantic assignment from out/debug/segments_0600.jpg:
attack_near = segs[4]
sideline_left = segs[14]
# right sideline detected in two pieces (18 near net, 15 lower) -> fit one line
p = np.array(segs[18]).reshape(2, 2), np.array(segs[15]).reshape(2, 2)
pts = np.vstack(p)
d = pts[-1] - pts[0]
sideline_right = (*pts[0], *(pts[0] + d))

anl = intersect(attack_near, sideline_left)
anr = intersect(attack_near, sideline_right)
print("attack_near_left", anl, "attack_near_right", anr)

ann = {
    # net-plane points, frame_0000 estimates shifted by measured camera motion (-48, -17)
    "net_top_left": [297, 251],
    "net_top_right": [940, 218],
    "antenna_tip_left": [296, 203],
    "antenna_tip_right": [938, 173],
    "attack_near_left": list(anl),
    "attack_near_right": list(anr),
}
json.dump(ann, open("out/annotations_0600.json", "w"), indent=1)

from farmclip.calibrate import run
c = run("out/annotations_0600.json", "out/debug/frame_0600.jpg")
print("err", round(c["err"], 1), "f", round(c["f"]))
print({k: round(v, 1) for k, v in c["per_point_err"].items()})

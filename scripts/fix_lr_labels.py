"""Canonicalise L/R keypoint naming in the court pose labels.

The pack's GT uses a *world-anchored* convention: `_left` keypoints sit at the
LARGER image-x in ~97% of frames. A minority of frames carry the inverted
convention (the "L/R drift" bug) -- e.g. 6 of 22 corner_near pairs.

Rather than fix pair-by-pair (which cannot resolve pairs where only one member
is visible), we detect orientation PER FRAME by majority vote over every pair
that has both members visible, then swap all 9 pairs in the frames that
disagree. Originals are backed up to court/labels_orig/.

Usage: python scripts/fix_lr_labels.py [--apply]
"""
import shutil
import sys
from pathlib import Path

import json

import numpy as np

ROOT = Path(__file__).parent.parent
COURT = ROOT / "court"
NAMES = json.loads((Path(__file__).parent.parent / "court/kpt_names.json").read_text())
PAIRS = [(i, i + 1) for i in range(0, 18, 2)]  # (left, right) index pairs
APPLY = "--apply" in sys.argv

label_files = sorted(COURT.glob("labels/*/*.txt"))
if not label_files:
    sys.exit("no label files found")

if APPLY:
    backup = COURT / "labels_orig"
    if backup.exists():
        sys.exit(f"{backup} already exists -- refusing to overwrite the backup")
    shutil.copytree(COURT / "labels", backup)
    print(f"backed up originals -> {backup}")

flipped, undecidable, per_pair, changed = [], [], [], 0
for lp in label_files:
    vals = [float(v) for v in lp.read_text().split()]
    head, kp = vals[:5], np.array(vals[5:]).reshape(18, 3)
    dirty = False

    agree = disagree = 0
    for a, b in PAIRS:
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            if kp[a, 0] > kp[b, 0]:
                agree += 1      # canonical convention: left at larger image-x
            else:
                disagree += 1
    if agree == disagree == 0:
        undecidable.append(lp.name)
    elif disagree > agree:
        # Whole frame carries the inverted convention -- flip every pair, which
        # also fixes pairs where only one member is visible.
        flipped.append((lp.name, agree, disagree))
        for a, b in PAIRS:
            kp[[a, b]] = kp[[b, a]]
        dirty = True

    # Residual per-pair errors: a single mislabelled pair in an otherwise
    # correctly-oriented frame. Only decidable when both members are visible.
    for a, b in PAIRS:
        if kp[a, 2] > 0 and kp[b, 2] > 0 and kp[a, 0] < kp[b, 0]:
            per_pair.append((lp.name, NAMES[a].replace("_left", "")))
            kp[[a, b]] = kp[[b, a]]
            dirty = True

    if dirty:
        changed += 1
    if APPLY and dirty:
        out = head + [x for row in kp for x in row]
        lp.write_text(" ".join(
            f"{int(v)}" if i == 0 else f"{v:.6g}"
            for i, v in enumerate(out)) + "\n")

print(f"\nscanned {len(label_files)} label files")
print(f"frames with inverted convention (whole-frame flip): {len(flipped)}")
for n, a, d in flipped:
    print(f"  {n:<28} agree={a} disagree={d}")
if per_pair:
    print(f"residual single-pair mislabels fixed: {len(per_pair)}")
    for n, pr in per_pair:
        print(f"  {n:<28} {pr}")
if undecidable:
    print(f"frames with no both-visible pair (orientation undecidable): {len(undecidable)}")
    for n in undecidable:
        print(f"  {n}")
print(f"\n{'APPLIED' if APPLY else 'DRY RUN'}: {changed} files "
      f"{'rewritten' if APPLY else 'would be rewritten'}")
if not APPLY:
    print("re-run with --apply to write changes")

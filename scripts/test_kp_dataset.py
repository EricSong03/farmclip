"""Sanity check the YOLO pose dataset from build_kp_dataset.py. Plain asserts."""
from pathlib import Path

OUT = Path(__file__).parent.parent / "data/dataset/court"

labels = sorted(OUT.glob("labels/*/*.txt"))[:2]
assert len(labels) == 2, f"need 2 label files, found {len(labels)}"
for p in labels:
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1, f"{p}: {len(lines)} lines"
    nums = [float(x) for x in lines[0].split()]
    assert len(nums) == 5 + 54, f"{p}: {len(nums)} numbers"
    assert nums[0] == 0
    coords = nums[1:5] + [v for i, v in enumerate(nums[5:]) if i % 3 != 2]
    assert all(0 <= v <= 1 for v in coords), f"{p}: coord out of [0,1]"
    vis = nums[5:][2::3]
    assert all(v in (0, 2) for v in vis), f"{p}: bad visibility {set(vis)}"

yaml_txt = (OUT / "dataset.yaml").read_text()
assert "kpt_shape: [18, 3]" in yaml_txt, "kpt_shape wrong"
print("ok")

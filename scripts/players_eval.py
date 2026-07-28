"""Before/after acceptance eval for the player student.

For each held-out (time-disjoint) menlo val frame, run BOTH the stock yolo11s and
the fine-tuned student at the runtime config (imgsz=1280, conf=0.2) and draw them
on one image: stock=BLUE, finetuned=RED. The acceptance question is visual and
specific — does the student now box the far-side players behind the net that stock
misses? Also prints per-frame person counts and stock->ft delta.

Runs on CPU EP by design (device='cpu') so counts reflect real runtime, not H200.
"""
from pathlib import Path
import cv2
from ultralytics import YOLO

VAL_DIR = Path("player_dataset/images/val")
OUT = Path("out/debug/players_eval")
OUT.mkdir(parents=True, exist_ok=True)
IMGSZ, CONF = 1280, 0.2

stock = YOLO("yolo11s.pt")
ft = YOLO("finetune_out/yolo11s-vb.pt")

# only menlo (clip_*) val frames — that's the hard far-side-occlusion class
frames = sorted(VAL_DIR.glob("clip_*.jpg"))
print(f"{len(frames)} menlo val frames")
tot_s = tot_f = 0
for fp in frames:
    img = cv2.imread(str(fp))
    rs = stock.predict(img, classes=[0], conf=CONF, imgsz=IMGSZ, device="cpu", verbose=False)[0]
    rf = ft.predict(img, classes=[0], conf=CONF, imgsz=IMGSZ, device="cpu", verbose=False)[0]
    vis = img.copy()
    for x1, y1, x2, y2 in rs.boxes.xyxy.tolist():  # stock = blue
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 128, 0), 2)
    for x1, y1, x2, y2 in rf.boxes.xyxy.tolist():  # finetuned = red
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    ns, nf = len(rs.boxes), len(rf.boxes)
    tot_s += ns; tot_f += nf
    cv2.putText(vis, f"{fp.stem}  stock(blue)={ns}  ft(red)={nf}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / f"{fp.stem}.jpg"), vis)
    print(f"{fp.stem}: stock={ns}  ft={nf}  delta={nf-ns:+d}")
print(f"TOTAL persons  stock={tot_s}  ft={tot_f}  delta={tot_f-tot_s:+d}")
print(f"overlays -> {OUT}")

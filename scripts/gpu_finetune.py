"""GPU fine-tune session for farmclip (run on ICRN H200, not locally).

Player detector: pseudo-label our clips with a big teacher (yolo11x @ 1536),
then fine-tune yolo11s @ 1280 as the CPU-runtime student. Ball fine-tune
(VballNet) uses the vendor repo's own trainer — see docs/plans/icrn-finetune.md.

Stages:
  python gpu_finetune.py players-data --videos videos/       # teacher pseudo-labels
  python gpu_finetune.py players-train                       # fine-tune yolo11s
Outputs land in finetune_out/.

Deps: pip install ultralytics opencv-python-headless
"""
import argparse
import shutil
from pathlib import Path

import cv2

OUT = Path("finetune_out")
DATA = Path("player_dataset")
TEACHER = "yolo11x.pt"
STUDENT = "yolo11s.pt"
SAMPLE_EVERY = 15   # frames between sampled training images
TEACHER_IMGSZ = 1536
TEACHER_CONF = 0.25
TRAIN_IMGSZ = 1280
EPOCHS = 60
VAL_FRAC = 0.1


def players_data(videos_dir):
    from ultralytics import YOLO
    model = YOLO(TEACHER)
    for split in ("train", "val"):
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_img = 0
    for vid in sorted(Path(videos_dir).glob("*.mp4")):
        cap = cv2.VideoCapture(str(vid))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # time-disjoint val: the temporally-LAST VAL_FRAC of each clip is held
        # out (never adjacent to a train frame), not a random 10% shuffle — so
        # the val score can't be inflated by frame-adjacency leakage.
        val_start = int(total * (1.0 - VAL_FRAC))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % SAMPLE_EVERY == 0:
                r = model.predict(frame, classes=[0], conf=TEACHER_CONF,
                                  imgsz=TEACHER_IMGSZ, verbose=False)[0]
                if len(r.boxes):
                    split = "val" if i >= val_start else "train"
                    stem = f"{vid.stem}_{i:06d}"
                    h, w = frame.shape[:2]
                    cv2.imwrite(str(DATA / "images" / split / f"{stem}.jpg"), frame)
                    lines = []
                    for b in r.boxes.xyxy.tolist():
                        x1, y1, x2, y2 = b
                        lines.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                                     f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
                    (DATA / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
                    n_img += 1
            i += 1
        cap.release()
        print(f"{vid.name}: sampled through frame {i}")
    (DATA / "data.yaml").write_text(
        f"path: {DATA.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\n")
    assert n_img > 100, f"only {n_img} pseudo-labeled images — check videos dir"
    print(f"dataset: {n_img} images in {DATA}")


def players_train():
    from ultralytics import YOLO
    model = YOLO(STUDENT)
    # crop/scale augs help far-side small players and edge-truncated boxes
    model.train(data=str(DATA / "data.yaml"), imgsz=TRAIN_IMGSZ, epochs=EPOCHS,
                scale=0.7, translate=0.2, degrees=0.0, mosaic=1.0,
                project=str(OUT), name="yolo11s-vb")
    best = OUT / "yolo11s-vb" / "weights" / "best.pt"
    shutil.copy(best, OUT / "yolo11s-vb.pt")
    print(f"student weights: {OUT / 'yolo11s-vb.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["players-data", "players-train"])
    ap.add_argument("--videos", default="videos")
    args = ap.parse_args()
    if args.stage == "players-data":
        players_data(args.videos)
    else:
        players_train()


if __name__ == "__main__":
    main()

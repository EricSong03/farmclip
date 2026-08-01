"""Train the court-line segmentation UNet (Phase 2, line-seg-calib plan).

Data: data/dataset/lineseg (build_lineseg_dataset.py). 9-class UNet, weighted
CE + soft dice. Checkpoints: finetune_out/lineseg/{last,best}.pt.
Usage: python -m uv run python scripts/train_lineseg.py [--steps 30 --batch 2 --device cpu]
Export: python -m uv run python scripts/train_lineseg.py --export model.onnx
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data/dataset/lineseg"
CKPT = ROOT / "finetune_out/lineseg"
N_CLASSES = 9


def block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
        nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, n=N_CLASSES):
        super().__init__()
        self.inc = block(3, 32)
        self.downs = nn.ModuleList([block(c, c2) for c, c2 in
                                    [(32, 64), (64, 128), (128, 256), (256, 256)]])
        self.ups = nn.ModuleList([block(c + s, o) for c, s, o in
                                  [(256, 256, 128), (128, 128, 64), (64, 64, 32), (32, 32, 32)]])
        self.head = nn.Conv2d(32, n, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for d in self.downs:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        x = skips.pop()
        for u in self.ups:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = u(torch.cat([x, skips.pop()], 1))
        return self.head(x)


class LineSegDS(Dataset):
    def __init__(self, split, size, train):
        self.imgs = sorted((DATA / "images" / split).glob("*.jpg"))
        self.size, self.train = size, train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = cv2.imread(str(p))
        mask = cv2.imread(str(DATA / "masks" / p.parent.name / (p.stem + ".png")),
                          cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        tw, th = self.size
        if self.train:
            # random crop at target aspect (scale 0.7-1.0 of max fit), then resize
            s = random.uniform(0.7, 1.0)
            cw = int(min(W, H * tw / th) * s)
            ch = int(cw * th / tw)
            x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
            img, mask = img[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]
        img = cv2.resize(img, (tw, th))
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        if self.train and random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
            # hflip mirrors world left/right: swap sideline classes only
            # (far/near are depth, unaffected; center/net symmetric)
            m = mask.copy()
            m[mask == 1], m[mask == 2] = 2, 1
            mask = m
        x = torch.from_numpy(np.ascontiguousarray(img[..., ::-1])).permute(2, 0, 1).float() / 255
        return x, torch.from_numpy(np.ascontiguousarray(mask)).long()


def loss_fn(logits, target, ce_w):
    ce = F.cross_entropy(logits, target, weight=ce_w)
    p = logits.softmax(1)[:, 1:]
    t = F.one_hot(target, N_CLASSES).permute(0, 3, 1, 2)[:, 1:].float()
    inter, denom = (p * t).sum((0, 2, 3)), p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    return ce + 1 - ((2 * inter + 1) / (denom + 1)).mean()


@torch.no_grad()
def validate(model, loader, ce_w, dev):
    model.eval()
    tot, n = 0.0, 0
    inter = torch.zeros(N_CLASSES)
    union = torch.zeros(N_CLASSES)
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)
        tot += loss_fn(logits, y, ce_w).item()
        n += 1
        pred = logits.argmax(1).cpu()
        yc = y.cpu()
        for c in range(N_CLASSES):
            inter[c] += ((pred == c) & (yc == c)).sum()
            union[c] += ((pred == c) | (yc == c)).sum()
    iou = (inter / union.clamp(min=1)).tolist()
    return tot / max(n, 1), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, nargs=2, default=[960, 544])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=0, help="stop after N optimizer steps (smoke)")
    ap.add_argument("--export", help="export best.pt to this onnx path and exit")
    a = ap.parse_args()

    CKPT.mkdir(parents=True, exist_ok=True)
    model = UNet()
    if a.export:
        ck = CKPT / "best.pt" if (CKPT / "best.pt").exists() else CKPT / "last.pt"
        model.load_state_dict(torch.load(ck, map_location="cpu")["model"])
        model.eval()
        torch.onnx.export(model, torch.zeros(1, 3, a.imgsz[1], a.imgsz[0]), a.export,
                          input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "b"}, "logits": {0: "b"}},
                          opset_version=17, dynamo=False)  # dynamo path needs onnxscript
        print(f"exported {ck.name} -> {a.export}")
        return

    dev = torch.device(a.device)
    model.to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M device: {dev}")
    tr = DataLoader(LineSegDS("train", a.imgsz, True), batch_size=a.batch,
                    shuffle=True, num_workers=2, drop_last=True)
    va = DataLoader(LineSegDS("val", a.imgsz, False), batch_size=a.batch, num_workers=2)
    ce_w = torch.tensor([0.1] + [1.0] * 8).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.steps or a.epochs * max(len(tr), 1))
    step = 0
    best = float("inf")
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y, ce_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
            step += 1
            if a.steps:
                print(f"step {step} loss {loss.item():.4f}")
                if step >= a.steps:
                    torch.save({"model": model.state_dict()}, CKPT / "last.pt")
                    print(f"smoke done: saved {CKPT / 'last.pt'}")
                    return
        vl, iou = validate(model, va, ce_w, dev)
        print(f"epoch {ep + 1}/{a.epochs} train {tot / max(n, 1):.4f} val {vl:.4f} "
              f"iou {' '.join(f'{v:.2f}' for v in iou[1:])}")
        torch.save({"model": model.state_dict()}, CKPT / "last.pt")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict()}, CKPT / "best.pt")


if __name__ == "__main__":
    main()

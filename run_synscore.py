"""
SynScore (Eq. 6): train each vision architecture from scratch to detect a pasted 60x60 yellow square,
then measure how well each attribution method localises the square (AP; AUC-ROC also stored).

Reproduces M4_XAI_Benchmark/training (train.sh, train_model.py, data.py) and benchmark-cv/evaluate_pgs.py:
  * train: the 10k ImageNet-train images of benchmark_data/imagenet_train_10k.txt;
    Resize(256) -> RandomHorizontalFlip -> RandomCrop(224) -> ToTensor -> Normalize(ImageNet), then with
    probability 0.5 the normalised square (2.2489, 2.4286, -1.8044) at a random position -> label 1, else 0.
    Batch 64, shuffled, 3 epochs, cross-entropy, constant learning rate (the PiecewiseDecay scheduler in
    train_model.py is never stepped): SGD-momentum 0.9 with lr 0.01 (ResNets), 0.001 (VGG16),
    0.1 (MobileNet); AdamW lr 1e-3 (ViTs, MAE-ViT). From scratch (pretrained=False) with Paddle's default
    initialisers. The model after the last epoch is kept.
  * eval: Resize(224) -> CenterCrop(224) on the first 2500 images of imagenet_val_5k.txt with the square on
    a 50x50 grid; explanation of the predicted class; AP / AUC against the square mask after resizing the
    explanation to 224x224 (cv2 bilinear). Both the "abs" and "square" channel reductions are scored.

    python run_synscore.py train --models all
    python run_synscore.py eval  --models all
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from m4bench.common import IMAGENET_MEAN, IMAGENET_STD, ScoreLog, get_device, load_image_list, seed_everything
from m4bench.vision_explainers import EXPLAINERS
from m4bench.vision_metrics import SQUARE, synscore, synthetic_val_position
from m4bench.vision_models import VISION_MODEL_NAMES, load_vision_model

LR = {"resnet50": 0.01, "resnet101": 0.01, "resnet152": 0.01, "vgg16": 0.001, "mobilenet": 0.1,
      "vit_small": 1e-3, "vit_base": 1e-3, "vit_large": 1e-3, "mae_vit_base": 1e-3}
N_SYN_VAL = 2500
YELLOW_NORMALISED = (2.2489, 2.4286, -1.8044)      # training/data.py


# ─────────────────────────────────────────────────────────────────────────────
# paddle.vision.transforms (PIL backend) semantics
# ─────────────────────────────────────────────────────────────────────────────

def paddle_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if (w <= h and w == size) or (h <= w and h == size):
        return img
    if w < h:
        ow, oh = size, int(size * h / w)
    else:
        oh, ow = size, int(size * w / h)
    return img.resize((ow, oh), Image.BILINEAR)


def paddle_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    i, j = int(round((h - size) / 2.0)), int(round((w - size) / 2.0))
    return img.crop((j, i, j + size, i + size))


def to_normalised(img: Image.Image) -> torch.Tensor:
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def paste(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    x = x.clone()
    for c, v in enumerate(YELLOW_NORMALISED):
        x[c, h:h + SQUARE, w:w + SQUARE] = v
    return x


def load_rgb(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


class YellowTrain(Dataset):
    """data.py Yellow(mode='random')."""

    def __init__(self, paths, seed):
        self.paths, self.seed, self.epoch = paths, seed, 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        img = paddle_resize(load_rgb(self.paths[i]), 256)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w, h = img.size
        top, left = int(rng.integers(0, h - 224 + 1)), int(rng.integers(0, w - 224 + 1))
        x = to_normalised(img.crop((left, top, left + 224, top + 224)))
        if rng.random() > 0.5:
            hh, ww = (rng.random(2) * (x.shape[1] - SQUARE)).astype(np.int32)
            return paste(x, int(hh), int(ww)), 1
        return x, 0


def val_sample(path, idx):
    """data.py Yellow(mode='val'): first 2500 patched on the grid (label 1), the rest clean (label 0)."""
    x = to_normalised(paddle_center_crop(paddle_resize(load_rgb(path), 224), 224))
    if idx < N_SYN_VAL:
        h, w = synthetic_val_position(idx)
        return paste(x, h, w), 1, (h, w)
    return x, 0, None


def paddle_default_init(net: nn.Module) -> None:
    """Paddle defaults: Conv N(0, sqrt(2 / fan_in)), Linear Xavier-uniform, biases 0, BN (1, 0)."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            fan_in = m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
            nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / fan_in))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def ckpt_path(root, name):
    return Path(root) / f"{name}_yellow.pt"


def train(name, args, device):
    vm = load_vision_model(name, device, num_classes=2, pretrained=False)
    if vm.family == "cnn":
        paddle_default_init(vm.net)            # the ViT mirrors already apply vit.py's _init_weights
    for p in vm.net.parameters():
        p.requires_grad_(True)
    vm.net.train()
    ds = YellowTrain(load_image_list(args.train_dir, args.train_list), args.seed)
    dl = DataLoader(ds, batch_size=args.micro_bs, shuffle=True, num_workers=args.workers)
    accum = max(1, args.bs // args.micro_bs)
    opt = (torch.optim.AdamW(vm.net.parameters(), lr=LR[name], weight_decay=0.01) if vm.family == "vit"
           else torch.optim.SGD(vm.net.parameters(), lr=LR[name], momentum=0.9))
    for epoch in range(args.epochs):
        ds.epoch = epoch
        t0, seen, correct, loss_sum = time.time(), 0, 0, 0.0
        opt.zero_grad()
        for step, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            logits = vm.net(x)
            loss = F.cross_entropy(logits, y)
            (loss * len(y) / args.bs).backward()             # mean over the effective batch of 64
            if (step + 1) % accum == 0 or step + 1 == len(dl):
                opt.step()
                opt.zero_grad()
            seen += len(y)
            correct += int((logits.argmax(1) == y).sum())
            loss_sum += float(loss) * len(y)
        print(f"[{name}] epoch {epoch + 1}/{args.epochs} loss {loss_sum / seen:.4f} "
              f"train-acc {correct / seen:.4f} ({time.time() - t0:.0f}s)", flush=True)

    vm.net.eval()
    val_paths = load_image_list(args.data_dir, args.data_list)
    correct = 0
    with torch.no_grad():
        for i, p in enumerate(val_paths):
            x, y, _ = val_sample(p, i)
            correct += int(int(vm.net(x[None].to(device)).argmax(1)) == y)
    acc = correct / len(val_paths)
    print(f"[{name}] synthetic val accuracy {acc:.4f}", flush=True)
    out = ckpt_path(args.ckpt_dir, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": vm.net.state_dict(), "val_acc": acc}, out)


def evaluate(name, args, device):
    ck = torch.load(ckpt_path(args.ckpt_dir, name), map_location="cpu", weights_only=True)
    vm = load_vision_model(name, device, num_classes=2, pretrained=False)
    vm.net.load_state_dict(ck["state_dict"])
    methods = args.methods or vm.methods
    val_paths = load_image_list(args.data_dir, args.data_list)
    n = min(N_SYN_VAL, args.limit or N_SYN_VAL)
    root = Path(args.out) / name
    fields = ["SynScore_AP", "SynScore_AUC", "SynScore_AP_sq", "SynScore_AUC_sq", "pred"]
    logs = {m: ScoreLog(root / m / "synscore.csv", fields) for m in methods}
    kw = dict(ig_steps=args.ig_steps, ig_trials=args.ig_trials, sg_samples=args.sg_samples,
              sg_noise=0.1, bt_steps=20, grad_bs=args.grad_bs)
    t0 = time.time()
    for i in range(n):
        if all(i in logs[m].done for m in methods):
            continue
        x, _, (h, w) = val_sample(val_paths[i], i)
        x = x[None].to(device)
        with torch.no_grad():
            label = int(vm.net(x).argmax(1))
        for mi, m in enumerate(methods):
            if i in logs[m].done:
                continue
            expl = EXPLAINERS[m](vm, x, label, np.random.default_rng((args.seed, i, mi)), **kw)
            s_abs, s_sq = synscore(expl["abs"], h, w), synscore(expl["sq"], h, w)
            logs[m].add(i, {**s_abs, "SynScore_AP_sq": s_sq["SynScore_AP"],
                            "SynScore_AUC_sq": s_sq["SynScore_AUC"], "pred": label})
        if (i + 1) % 50 == 0:
            print(f"  [{name}] {i + 1}/{n} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    (root / "synthetic_model.json").write_text(json.dumps({"val_acc": ck["val_acc"]}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["train", "eval"])
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--train_dir", default="data/imagenet_train_10k")
    p.add_argument("--train_list", default="benchmark_data/imagenet_train_10k.txt")
    p.add_argument("--data_dir", default="data/M4_5000")
    p.add_argument("--data_list", default="benchmark_data/imagenet_val_5k.txt")
    p.add_argument("--ckpt_dir", default="models/synthetic")
    p.add_argument("--out", default="results/m4/vision")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=64, help="effective batch size (train_model.py)")
    p.add_argument("--micro_bs", type=int, default=64,
                   help="per-step batch, accumulated to --bs. Exact for VGG16 and the ViTs (no BatchNorm); "
                        "for ResNets/MobileNet anything below 64 changes BatchNorm batch statistics")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--ig_trials", type=int, default=10)
    p.add_argument("--sg_samples", type=int, default=100)
    p.add_argument("--grad_bs", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)
    device = get_device()
    for name in VISION_MODEL_NAMES if args.models == ["all"] else args.models:
        (train if args.stage == "train" else evaluate)(name, args, device)


if __name__ == "__main__":
    main()

"""Shared helpers: seeding, image I/O matching InterpretDL, resumable per-sample result stores."""

import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MASK_VALUE = 127          # InterpretDL Perturbation / Infidelity fill value, uint8 space


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────

def read_image_uint8(path: str, resize_to: int = 224, crop_to: int | None = 224) -> np.ndarray:
    """(H, W, 3) uint8 RGB. InterpretDL read_image: short side -> resize_to (cv2 linear), centre crop."""
    with open(path, "rb") as f:
        img = np.array(Image.open(f).convert("RGB"))
    h, w = img.shape[:2]
    scale = resize_to / min(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    if crop_to is None:
        return img
    h, w = img.shape[:2]
    top, left = (h - crop_to) // 2, (w - crop_to) // 2
    return np.ascontiguousarray(img[top:top + crop_to, left:left + crop_to])


def to_tensor01(imgs_nhwc: np.ndarray, device) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(imgs_nhwc)).to(device)
    return t.permute(0, 3, 1, 2).float().div_(255.0)


def load_image_list(data_dir: str, data_list: str | None, limit: int | None = None):
    """
    Image paths in the authors' order (benchmark_data/imagenet_val_5k.txt) when the list is given,
    otherwise every image under data_dir sorted. Order matters for SynScore (first 2500 get patches).
    """
    if data_list:
        with open(data_list) as f:
            rel = [l.strip() for l in f if l.strip()]
        paths = [os.path.join(data_dir, r) for r in rel]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(paths)} images from {data_list} are missing under {data_dir}, "
                f"e.g. {missing[0]}")
    else:
        exts = {".jpeg", ".jpg", ".png"}
        paths = sorted(str(p) for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts)
        if not paths:
            raise FileNotFoundError(f"no images found under {data_dir}")
    return paths[:limit] if limit else paths


# ─────────────────────────────────────────────────────────────────────────────
# Resumable result storage
# ─────────────────────────────────────────────────────────────────────────────

class ScoreLog:
    """Append-only per-sample CSV so long runs can be resumed: rows already written are skipped."""

    def __init__(self, path: Path, fields: list[str]):
        self.path = Path(path)
        self.fields = ["idx"] + fields
        self.done = set()
        if self.path.exists():
            with open(self.path) as f:
                self.done = {int(r["idx"]) for r in csv.DictReader(f)}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.fields)

    def add(self, idx: int, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([idx] + [row[k] for k in self.fields[1:]])
        self.done.add(idx)


class MapStore:
    """
    float32 memmap of raw 2-D explanation maps (N, H, W) at the method's native resolution, kept for
    PScore, which needs every model's map of the same image at once.
    """

    def __init__(self, path: Path, n: int, hw: tuple[int, int]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+" if self.path.exists() else "w+"
        self.arr = np.lib.format.open_memmap(self.path, mode=mode, dtype=np.float32, shape=(n, *hw))

    def put(self, idx: int, m: np.ndarray) -> None:
        self.arr[idx] = m.astype(np.float32)

    def flush(self) -> None:
        self.arr.flush()

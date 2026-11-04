"""
Vision faithfulness metrics, ported from InterpretDL evaluate_interpreter/{perturbation,infidelity}.py
and evaluate_models/localization.py (PointGameSegmentation) as configured by the M4 scripts.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import MASK_VALUE, to_tensor01
from .vision_models import VisionModel


def resize_map(expl: np.ndarray, hw=(224, 224)) -> np.ndarray:
    expl = np.asarray(expl, dtype=np.float32)
    if expl.shape != hw:
        expl = cv2.resize(expl, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return expl


@torch.no_grad()
def batched_outputs(vm: VisionModel, imgs_nhwc: np.ndarray, device, bs: int, amp: bool,
                    kind: str = "proba") -> torch.Tensor:
    outs = []
    for i in range(0, len(imgs_nhwc), bs):
        x01 = to_tensor01(imgs_nhwc[i:i + bs], device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits = vm.logits_from01(x01).float()
        outs.append(F.softmax(logits, 1) if kind == "proba" else logits)
    return torch.cat(outs)


# ─────────────────────────────────────────────────────────────────────────────
# MoRF / LeRF / ABPC  — Eq. (1)-(3)
# ─────────────────────────────────────────────────────────────────────────────

def perturbation_images(img: np.ndarray, expl: np.ndarray, n_steps: int):
    """
    Perturbation.generate_samples_array: q = 100/L, qs = [q(L-1), ..., q, 0];
    MoRF masks expl > p for p in percentiles(qs), LeRF masks expl < p for p reversed, both cumulative,
    with (127,127,127). Step 0 is the clean image, so each curve has L+1 points.
    M4 evaluates with limit_number_generated_samples=100 (benchmark-cv/run_eval.sh).
    """
    expl = resize_map(expl, img.shape[:2])
    q = 100.0 / n_steps
    thrs = np.percentile(expl, [q * (i - 1) for i in range(n_steps, 0, -1)])
    morf, lerf = [img], [img]
    cur = img.copy()
    for p in thrs:
        cur = cur.copy()
        cur[expl > p] = MASK_VALUE
        morf.append(cur)
    cur = img.copy()
    for p in thrs[::-1]:
        cur = cur.copy()
        cur[expl < p] = MASK_VALUE
        lerf.append(cur)
    return np.stack(morf), np.stack(lerf)


def morf_lerf_abpc(vm, img, expl, device, n_steps=100, bs=64, amp=False):
    morf_imgs, lerf_imgs = perturbation_images(img, expl, n_steps)
    probs = batched_outputs(vm, np.concatenate([morf_imgs, lerf_imgs]), device, bs, amp)
    coi = int(probs[0].argmax())                     # class of interest: top class of the clean image
    p = probs[:, coi].cpu().numpy()
    morf_p, lerf_p = p[:n_steps + 1], p[n_steps + 1:]
    return {
        "MoRF": float(morf_p[0] - morf_p.mean()),
        "LeRF": float(lerf_p[0] - lerf_p.mean()),
        "ABPC": float((lerf_p - morf_p).mean()),
    }, morf_p, lerf_p


# ─────────────────────────────────────────────────────────────────────────────
# INFD — Eq. (4)
# ─────────────────────────────────────────────────────────────────────────────

INFD_KERNELS = (32, 64, 128)
INFD_STRIDE = 8


def infd_boxes(h=224, w=224, kernels=INFD_KERNELS, stride=INFD_STRIDE):
    return [(y, x, k) for k in kernels
            for y in range(0, h - k + 1, stride)
            for x in range(0, w - k + 1, stride)]                 # 625 + 441 + 169 = 1235 for 224x224


class InfidelityCache:
    """
    Infidelity._generate_samples perturbs square patches (32/64/128 px, stride 8) to 127 and uses the
    logit of the clean image's top class. f(x) - f(x - I) depends only on (model, image), so it is
    computed once per image and reused for every explanation method, as upstream does (recompute=False).
    """

    def __init__(self, vm, img, device, bs=64, amp=False):
        self.boxes = infd_boxes(*img.shape[:2])
        logit0 = batched_outputs(vm, img[None], device, 1, amp, kind="logit")[0]
        self.label = int(logit0.argmax())
        diffs = []
        for i in range(0, len(self.boxes), bs):
            batch = np.repeat(img[None], len(self.boxes[i:i + bs]), axis=0)
            for j, (y, x, k) in enumerate(self.boxes[i:i + bs]):
                batch[j, y:y + k, x:x + k] = MASK_VALUE
            out = batched_outputs(vm, batch, device, bs, amp, kind="logit")[:, self.label]
            diffs.append((logit0[self.label] - out).cpu().numpy())
        self.diff = np.concatenate(diffs).astype(np.float64)

    def score(self, expl: np.ndarray) -> float:
        e = resize_map(expl).astype(np.float64)
        ii = np.pad(e.cumsum(0).cumsum(1), ((1, 0), (1, 0)))          # integral image -> box sums in O(1)
        exp_sum = np.array([ii[y + k, x + k] - ii[y, x + k] - ii[y + k, x] + ii[y, x]
                            for (y, x, k) in self.boxes])
        denom = np.mean(exp_sum * exp_sum)
        if denom == 0.0:
            exp_sum = np.zeros_like(exp_sum)
        else:
            exp_sum = exp_sum * (np.mean(self.diff * exp_sum) / denom)   # optimal scaling beta
        return float(np.mean(np.square(self.diff - exp_sum)))


# ─────────────────────────────────────────────────────────────────────────────
# SynScore — Eq. (6)
# ─────────────────────────────────────────────────────────────────────────────

SQUARE = 60
YELLOW = (255, 255, 0)      # the normalised (2.2489, 2.4286, -1.8044) of training/data.py in pixel space


def synthetic_val_position(idx: int, size: int = 224, square: int = SQUARE):
    """training/data.py Yellow(mode='val'): the first 2500 images get a patch on a 50x50 grid."""
    stride = (size - square) / 50
    return int((idx // 50) * stride), int((idx % 50) * stride)


def paste_square(img: np.ndarray, h: int, w: int, square: int = SQUARE) -> np.ndarray:
    out = img.copy()
    out[h:h + square, w:w + square] = YELLOW
    return out


def synscore(expl: np.ndarray, h: int, w: int, size: int = 224, square: int = SQUARE) -> dict:
    gt = np.zeros((size, size), np.uint8)
    gt[h:h + square, w:w + square] = 1
    e = resize_map(expl, (size, size)).ravel()
    return {"SynScore_AP": float(average_precision_score(gt.ravel(), e)),
            "SynScore_AUC": float(roc_auc_score(gt.ravel(), e))}

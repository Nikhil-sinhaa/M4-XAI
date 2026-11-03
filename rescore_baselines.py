"""
Phase 2 extension: re-score the saved M4 attribution maps under different masking baselines.

The expensive part of M4 is computing the attributions (IG = 500 fwd+bwd passes per image). Those
maps are already stored in results/m4/vision/<model>/<method>/maps_sq.npy, so evaluating a new
masking baseline only needs forward passes: no explainer is re-run.

For every (model, method, baseline) this stores the MoRF and LeRF class-probability curves
(N images x 101 points), from which MoRF / LeRF / ABPC are derived exactly as in
m4bench.vision_metrics.morf_lerf_abpc. Analysis lives in analyze_baselines.py.

    python rescore_baselines.py --models resnet50 --limit 500
    python rescore_baselines.py --models resnet50 --methods ig --limit 20 --out /tmp/check   # quick test

Baselines (the grey one is the M4 default and is the "before" reference):
    grey   (127,127,127)                       M4 / InterpretDL default
    black  (0,0,0)
    mean   ImageNet mean colour
    blur   Gaussian-blurred copy of the image  (stays close to the data distribution)
    noise  uniform per-pixel noise             (seeded per image)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from m4bench.common import get_device, load_image_list, read_image_uint8, seed_everything, IMAGENET_MEAN
from m4bench.vision_metrics import batched_outputs, resize_map
from m4bench.vision_models import VISION_MODEL_NAMES, load_vision_model

BASELINES = ["grey", "black", "mean", "blur", "noise"]
BLUR_SIGMA = 11.0       # same strength as phase2_vision_baseline.py (PIL GaussianBlur radius 11)


def baseline_image(name: str, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The (H, W, 3) uint8 image that masked pixels are replaced with."""
    if name == "grey":
        return np.full_like(img, 127)
    if name == "black":
        return np.zeros_like(img)
    if name == "mean":
        return np.broadcast_to(np.round(np.array(IMAGENET_MEAN) * 255).astype(np.uint8), img.shape).copy()
    if name == "blur":
        return cv2.GaussianBlur(img, (0, 0), BLUR_SIGMA)
    if name == "noise":
        return rng.integers(0, 256, size=img.shape, dtype=np.uint8)
    raise ValueError(name)


def perturbation_stacks(img, expl, base, n_steps):
    """
    Same schedule as m4bench.vision_metrics.perturbation_images, but masked pixels take the value of
    `base` instead of a constant. The masks are nested, so each step is just the largest mask so far.
    Returns two (n_steps + 1, H, W, 3) uint8 stacks; step 0 is the clean image.
    """
    expl = resize_map(expl, img.shape[:2])
    q = 100.0 / n_steps
    thrs = np.percentile(expl, [q * (i - 1) for i in range(n_steps, 0, -1)])
    morf = np.stack([img] + [np.where((expl > p)[..., None], base, img) for p in thrs])
    lerf = np.stack([img] + [np.where((expl < p)[..., None], base, img) for p in thrs[::-1]])
    return morf, lerf


def rescore_method(vm, paths, maps, baselines, device, n_steps, bs, amp, seed, idx_list):
    """-> (len(baselines), N, 2, n_steps + 1) float16 curves: [:, :, 0] MoRF probs, [:, :, 1] LeRF probs."""
    out = np.zeros((len(baselines), len(idx_list), 2, n_steps + 1), np.float16)
    for row, idx in enumerate(idx_list):
        img = read_image_uint8(paths[idx])
        for b, name in enumerate(baselines):
            rng = np.random.default_rng((seed, idx, b))
            morf, lerf = perturbation_stacks(img, maps[idx], baseline_image(name, img, rng), n_steps)
            probs = batched_outputs(vm, np.concatenate([morf, lerf]), device, bs, amp)
            coi = int(probs[0].argmax())                       # top class of the clean image, as in M4
            p = probs[:, coi].cpu().numpy()
            out[b, row, 0], out[b, row, 1] = p[:n_steps + 1], p[n_steps + 1:]
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["resnet50"], help=f"any of {VISION_MODEL_NAMES}")
    p.add_argument("--methods", nargs="+", default=None, help="default: every method found for the model")
    p.add_argument("--baselines", nargs="+", default=BASELINES, choices=BASELINES)
    p.add_argument("--maps_root", default="results/m4/vision")
    p.add_argument("--out", default="results/m4/rescore")
    p.add_argument("--data_dir", default="data/M4_5000")
    p.add_argument("--data_list", default="benchmark_data/imagenet_val_5k.txt")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--n_steps", type=int, default=100)
    p.add_argument("--bs", type=int, default=101, help="forward-pass batch size")
    p.add_argument("--amp", action="store_true", help="bf16 forward passes (faster, not bit-exact)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    paths = load_image_list(args.data_dir, args.data_list)
    idx_list = list(range(min(args.limit, len(paths))))
    for name in args.models:
        model_dir = Path(args.maps_root) / name
        methods = args.methods or sorted(d.name for d in model_dir.iterdir()
                                         if (d / "maps_sq.npy").exists() and d.name not in
                                         ("constant", "random", "random_16"))
        vm = load_vision_model(name, device)
        for m in methods:
            dst = Path(args.out) / name / f"{m}.npy"
            if dst.exists() and np.load(dst, mmap_mode="r").shape[1] >= len(idx_list):
                print(f"[{name}/{m}] already done, skipping", flush=True)
                continue
            maps = np.load(model_dir / m / "maps_sq.npy", mmap_mode="r")
            t0 = time.time()
            curves = rescore_method(vm, paths, maps, args.baselines, device, args.n_steps, args.bs,
                                    args.amp, args.seed, idx_list)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, curves)
            (dst.parent / "baselines.txt").write_text("\n".join(args.baselines))
            print(f"[{name}/{m}] {len(idx_list)} images x {len(args.baselines)} baselines "
                  f"in {(time.time() - t0) / 60:.1f} min -> {dst}", flush=True)
        del vm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

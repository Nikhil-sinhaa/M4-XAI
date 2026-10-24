"""
M4 vision benchmark: explain + evaluate (MoRF, LeRF, ABPC, INFD) for every model/method pair.

    python run_vision.py --models all                      # all 9 models, their M4 methods
    python run_vision.py --models resnet50 --baselines     # + Constant / Random-16 / Random (Table 1)
    python run_vision.py --models vit_base --limit 100     # quick check on 100 images

Outputs (resumable, re-running skips finished images):
    results/m4/vision/<model>/<method>/scores.csv   per-image MoRF, LeRF, ABPC, INFD (abs and square reductions)
    results/m4/vision/<model>/<method>/maps_sq.npy  native-resolution maps for PScore (compute_pscore.py)
    results/m4/vision/<model>/<method>/curves.npy   per-image MoRF / LeRF probability curves
    results/m4/vision/<model>/info.json             settings + top-1 accuracy on the evaluated images
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from m4bench.common import (MapStore, ScoreLog, get_device, load_image_list, read_image_uint8,
                            seed_everything, to_tensor01)
from m4bench.vision_explainers import EXPLAINERS, NATIVE_HW
from m4bench.vision_metrics import InfidelityCache, morf_lerf_abpc
from m4bench.vision_models import BASELINE_METHODS, VISION_MODEL_NAMES, load_vision_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"], help=f"'all' or any of {VISION_MODEL_NAMES}")
    p.add_argument("--methods", nargs="+", default=None,
                   help="override the per-model M4 method set (CNN: gradcam ig sg; ViT: ig sg ga bt_head bt_token)")
    p.add_argument("--baselines", action="store_true", help="also run constant / random_16 / random")
    p.add_argument("--data_dir", default="data/M4_5000", help="root holding <synset>/<file>.JPEG")
    p.add_argument("--data_list", default="benchmark_data/imagenet_val_5k.txt",
                   help="authors' list (fixes image order); pass '' to glob data_dir instead")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    p.add_argument("--out", default="results/m4/vision")
    # Perturbation / attribution settings (defaults = M4 scripts)
    p.add_argument("--n_steps", type=int, default=100, help="perturbation percentiles (run_eval.sh: 100)")
    p.add_argument("--ig_steps", type=int, default=50)
    p.add_argument("--ig_trials", type=int, default=10)
    p.add_argument("--sg_samples", type=int, default=100)
    p.add_argument("--sg_noise", type=float, default=0.1)
    p.add_argument("--bt_steps", type=int, default=20)
    p.add_argument("--start_layers", nargs="+", type=int, default=None,
                   help="GA/BT start-layer sweep, saved as e.g. ga-2 (upstream swept 2 4 6 8 and the default 3)")
    p.add_argument("--no_infd", action="store_true", help="skip INFD (1235 extra forward passes per image)")
    # Throughput
    p.add_argument("--bs", type=int, default=64, help="batch size for no-grad forward passes")
    p.add_argument("--grad_bs", type=int, default=25, help="batch size for gradient passes (lower for ViT-L)")
    p.add_argument("--amp", action="store_true", help="bf16 autocast for metric forward passes (faster, not exact)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_model(name, paths, synsets, args, device, total):
    vm = load_vision_model(name, device)
    base_methods = list(args.methods or vm.methods)
    if args.baselines:
        base_methods = BASELINE_METHODS + base_methods
    unsupported = [m for m in base_methods if m in ("gradcam",) and vm.family != "cnn"
                   or m in ("ga", "bt_head", "bt_token") and vm.family != "vit"]
    if unsupported:
        raise SystemExit(f"{name}: {unsupported} not applicable to a {vm.family} (paper leaves these blank)")
    # variant name -> (explainer, start layer); GA/BT get one variant per swept start layer
    variants = {}
    for m in base_methods:
        if m in ("ga", "bt_head", "bt_token") and args.start_layers:
            variants.update({f"{m}-{sl}": (m, sl) for sl in args.start_layers})
        else:
            variants[m] = (m, vm.start_layer)
    methods = list(variants)

    out = Path(args.out) / name
    n = len(paths)
    logs = {m: ScoreLog(out / m / "scores.csv", ["MoRF", "LeRF", "ABPC", "INFD", "INFD_sq"]) for m in methods}
    # Stores are sized for the full image list so a --limit run can later be extended to all 5000 images.
    maps = {m: MapStore(out / m / "maps_sq.npy", total, NATIVE_HW[variants[m][0]]) for m in methods}
    curves = {m: np.lib.format.open_memmap(out / m / "curves.npy", dtype=np.float16, shape=(total, 2, args.n_steps + 1),
                                           mode="r+" if (out / m / "curves.npy").exists() else "w+")
              for m in methods}
    kw = dict(ig_steps=args.ig_steps, ig_trials=args.ig_trials, sg_samples=args.sg_samples,
              sg_noise=args.sg_noise, bt_steps=args.bt_steps, grad_bs=args.grad_bs)

    correct = 0
    t0 = time.time()
    todo = [i for i in range(n) if any(i not in logs[m].done for m in methods)]
    print(f"[{name}] methods={methods} images={n} remaining={len(todo)}", flush=True)
    for step, idx in enumerate(todo):
        img = read_image_uint8(paths[idx])
        x = vm.normalize(to_tensor01(img[None], device))
        with torch.no_grad():
            label = int(vm.net(x).argmax(1))
        correct += int(synsets is not None and label == synsets[idx])
        infd = None if args.no_infd else InfidelityCache(vm, img, device, args.bs, args.amp)

        for mi, m in enumerate(methods):
            if idx in logs[m].done:
                continue
            rng = np.random.default_rng((args.seed, idx, mi))
            explainer, vm.start_layer = variants[m]
            expl = EXPLAINERS[explainer](vm, x, label, rng, **kw)
            scores, morf_p, lerf_p = morf_lerf_abpc(vm, img, expl["abs"], device, args.n_steps, args.bs, args.amp)
            # evaluate_2.py scores both channel reductions ("-abs" and "-square")
            scores["INFD"] = infd.score(expl["abs"]) if infd else float("nan")
            scores["INFD_sq"] = infd.score(expl["sq"]) if infd else float("nan")
            maps[m].put(idx, expl["sq"])
            curves[m][idx, 0], curves[m][idx, 1] = morf_p, lerf_p
            logs[m].add(idx, scores)

        if (step + 1) % 25 == 0 or step + 1 == len(todo):
            for m in methods:
                maps[m].flush()
                curves[m].flush()
            rate = (step + 1) / (time.time() - t0)
            print(f"  [{name}] {step + 1}/{len(todo)}  {rate * 3600:.0f} img/h  "
                  f"ETA {(len(todo) - step - 1) / rate / 3600:.1f} h", flush=True)

    info = {"model": name, "methods": {v: {"explainer": e, "start_layer": sl} for v, (e, sl) in variants.items()},
            "n_images": n,
            "settings": {k: getattr(args, k) for k in ("n_steps", "ig_steps", "ig_trials", "sg_samples",
                                                         "sg_noise", "bt_steps", "amp", "seed")}}
    if synsets is not None and todo:
        info["top1_on_newly_evaluated"] = correct / len(todo)
    info_path = out / "info.json"
    if info_path.exists():
        info = {**json.loads(info_path.read_text()), **info}
    info_path.write_text(json.dumps(info, indent=2))
    del vm
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    all_paths = load_image_list(args.data_dir, args.data_list or None)
    paths = all_paths[:args.limit] if args.limit else all_paths
    # torchvision's class index order is the sorted list of the 1000 synsets.
    wnids = sorted({Path(p).parent.name for p in all_paths})
    synsets = [wnids.index(Path(p).parent.name) for p in paths] if len(wnids) == 1000 else None
    models = VISION_MODEL_NAMES if args.models == ["all"] else args.models
    print(f"device={device} images={len(paths)} models={models}")
    for name in models:
        run_model(name, paths, synsets, args, device, len(all_paths))


if __name__ == "__main__":
    main()

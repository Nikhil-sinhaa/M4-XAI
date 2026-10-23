# -*- coding: utf-8 -*-
import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Phase 2: Baseline Sensitivity Analysis — Vision Modality
=========================================================
Research Gap: The M4 paper (NeurIPS 2023) uses a fixed grey-pixel masking
baseline (127,127,127) for perturbation-based faithfulness evaluation. This
script tests whether MoRF and ABPC scores are sensitive to the choice of
masking baseline for ResNet-50 on the ImageNet M4_5000 subset.

Baselines tested:
  1. grey      — (127,127,127) original M4 paper baseline
  2. black     — (0,0,0) zero-pixel masking
  3. blur      — Gaussian-blurred version of the image (in-distribution)
  4. noise     — Random Gaussian noise (OOD)

Attribution method: Integrated Gradients (best CNN performer per M4 paper)

Output:
  results/phase2/vision_baseline_results.csv
  results/phase2/vision_morf_curves.png
  results/phase2/vision_abpc_comparison.png
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("[error] captum not found — please install: pip install captum")
    exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (matching InterpretDL / M4 paper pipeline)
# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
N_STEPS       = 20   # perturbation steps (same as M4 paper)

BASELINES = ["grey", "black", "blur", "noise"]
BASELINE_LABELS = {
    "grey":  "Grey (127,127,127) [M4 Paper]",
    "black": "Black (0,0,0)",
    "blur":  "Gaussian Blur (in-dist.)",
    "noise": "Random Noise (OOD)",
}
BASELINE_COLORS = {
    "grey":  "#E63946",   # red
    "black": "#457B9D",   # blue
    "blur":  "#2A9D8F",   # teal
    "noise": "#E9C46A",   # yellow
}


# ──────────────────────────────────────────────────────────────────────────────
# Image loading (matching M4 / InterpretDL pipeline)
# ──────────────────────────────────────────────────────────────────────────────

def read_image_uint8(img_path: str, size: int = 224) -> np.ndarray:
    """Returns (H, W, 3) uint8 numpy RGB."""
    with open(img_path, "rb") as f:
        img = Image.open(f).convert("RGB")
    # Resize short side to `size`, then center-crop
    w, h = img.size
    if h < w:
        new_h, new_w = size, max(1, int(w * size / h))
    else:
        new_h, new_w = max(1, int(h * size / w)), size
    img = img.resize((new_w, new_h), Image.BILINEAR)
    top  = (new_h - size) // 2
    left = (new_w - size) // 2
    img  = img.crop((left, top, left + size, top + size))
    return np.array(img, dtype=np.uint8)


def normalize_image(img_hwc: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → (1,3,H,W) float32 normalized tensor."""
    x = img_hwc.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Masking Baseline Generators
# ──────────────────────────────────────────────────────────────────────────────

def make_masked_image(img_hwc: np.ndarray, mask_hw: np.ndarray,
                      baseline: str, blur_radius: int = 11) -> np.ndarray:
    """
    Given a boolean mask (True = pixels to be replaced), fill those pixels
    with the chosen baseline strategy.

    Args:
        img_hwc   : (H, W, 3) uint8 original image
        mask_hw   : (H, W) bool — True where pixels should be replaced
        baseline  : one of 'grey', 'black', 'blur', 'noise'
        blur_radius: kernel size for Gaussian blur baseline

    Returns:
        (H, W, 3) uint8 perturbed image
    """
    out = img_hwc.copy()

    if baseline == "grey":
        out[mask_hw] = 127

    elif baseline == "black":
        out[mask_hw] = 0

    elif baseline == "blur":
        # Gaussian blur of the original image — more in-distribution than grey
        pil_img = Image.fromarray(img_hwc)
        blurred = np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
                           dtype=np.uint8)
        out[mask_hw] = blurred[mask_hw]

    elif baseline == "noise":
        # Random Gaussian noise in [0,255]
        rng   = np.random.default_rng(seed=42)
        noise = rng.integers(0, 256, size=img_hwc.shape, dtype=np.uint8)
        out[mask_hw] = noise[mask_hw]

    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    return out


# ──────────────────────────────────────────────────────────────────────────────
# MoRF / LeRF image stack generation (exact M4 / InterpretDL formula)
# ──────────────────────────────────────────────────────────────────────────────

def generate_perturbation_stacks(
    img_hwc: np.ndarray,
    attr_map: np.ndarray,
    baseline: str,
    n_steps: int = N_STEPS,
) -> tuple:
    """
    Returns:
        morf_stack : list of (H, W, 3) uint8 — length n_steps+1, step 0 = original
        lerf_stack : list of (H, W, 3) uint8 — length n_steps+1
    """
    q    = 100.0 / n_steps
    qs   = [q * (n_steps - 1 - i) for i in range(n_steps)]  # [95, 90, …, 0]
    thrs = np.percentile(attr_map, qs)

    # MoRF — cumulative most-relevant-first masking
    morf_list = [img_hwc.copy()]
    fudged    = img_hwc.copy()
    for p in thrs:          # descending thresholds
        mask   = attr_map > p
        fudged = make_masked_image(fudged, mask, baseline)
        morf_list.append(fudged)

    # LeRF — cumulative least-relevant-first masking
    lerf_list = [img_hwc.copy()]
    fudged    = img_hwc.copy()
    for p in thrs[::-1]:    # ascending thresholds
        mask   = attr_map < p
        fudged = make_masked_image(fudged, mask, baseline)
        lerf_list.append(fudged)

    return morf_list, lerf_list


# ──────────────────────────────────────────────────────────────────────────────
# Model inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_model_on_stack(model, img_list, pred_cls, device) -> np.ndarray:
    """
    img_list: list of (H,W,3) uint8 images.
    Returns probabilities of pred_cls for each image — shape (len(img_list),).
    """
    probs = []
    for img_hwc in img_list:
        t = normalize_image(img_hwc).to(device)
        p = F.softmax(model(t), dim=1)[0, pred_cls].item()
        probs.append(p)
    return np.array(probs, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Attribution: Integrated Gradients
# ──────────────────────────────────────────────────────────────────────────────

def compute_ig_attribution(model, inp_norm: torch.Tensor, pred_cls: int,
                            device, ig_steps: int = 50) -> np.ndarray:
    """
    Returns pixel-level (H, W) float32 attribution map (sum of |attr| over channels).
    Uses IG with zero baseline (as in the simpler evaluate_vgg16.py baseline).
    """
    ig       = IntegratedGradients(model)
    baseline = torch.zeros_like(inp_norm).to(device)
    attr     = ig.attribute(inp_norm, baselines=baseline, target=pred_cls,
                            n_steps=ig_steps, internal_batch_size=1)
    torch.cuda.empty_cache()
    # Aggregate channel dimension: |attr| → sum over C
    a = attr.detach().cpu().numpy().squeeze(0)   # (3, H, W)
    a = np.abs(a).sum(axis=0)                     # (H, W)
    return a.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_image(model, img_path: str, device, ig_steps: int = 50):
    """
    Returns dict: baseline → {'morf_curve': np.array, 'lerf_curve': np.array,
                               'morf_score': float, 'abpc_score': float}
    Returns None if image cannot be processed.
    """
    try:
        img_hwc = read_image_uint8(img_path)
    except Exception as e:
        print(f"  [skip] {img_path}: {e}")
        return None

    inp_norm = normalize_image(img_hwc).to(device)

    with torch.no_grad():
        pred_cls = int(torch.argmax(model(inp_norm), dim=1).item())

    # Compute attribution once (shared across baselines)
    try:
        attr_map = compute_ig_attribution(model, inp_norm, pred_cls, device, ig_steps)
    except Exception as e:
        print(f"  [warn] IG failed on {img_path}: {e}")
        return None

    results = {}
    for baseline in BASELINES:
        morf_list, lerf_list = generate_perturbation_stacks(img_hwc, attr_map, baseline)

        morf_probs = run_model_on_stack(model, morf_list, pred_cls, device)
        lerf_probs = run_model_on_stack(model, lerf_list, pred_cls, device)

        # MoRF score: mean probability drop (M4 paper Eq. 1)
        morf_score = float((morf_probs[0] - morf_probs).mean())
        # ABPC score: mean(lerf_probs - morf_probs)
        abpc_score = float((lerf_probs - morf_probs).mean())

        results[baseline] = {
            "morf_curve": morf_probs,
            "lerf_curve": lerf_probs,
            "morf_score": morf_score,
            "abpc_score": abpc_score,
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_morf_curves(mean_curves: dict, n_steps: int, output_path: str):
    """Plot mean MoRF curves for each baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.linspace(0, 1, n_steps + 1)

    ax = axes[0]
    for baseline in BASELINES:
        curve = mean_curves[baseline]["morf"]
        ax.plot(x, curve, label=BASELINE_LABELS[baseline],
                color=BASELINE_COLORS[baseline], linewidth=2.2, marker='o',
                markersize=4)
    ax.set_xlabel("Fraction of Pixels Masked (MoRF)", fontsize=12)
    ax.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax.set_title("MoRF Curves by Masking Baseline\n(ResNet-50, Integrated Gradients)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    for baseline in BASELINES:
        curve = mean_curves[baseline]["lerf"]
        ax.plot(x, curve, label=BASELINE_LABELS[baseline],
                color=BASELINE_COLORS[baseline], linewidth=2.2, marker='s',
                markersize=4, linestyle='--')
    ax.set_xlabel("Fraction of Pixels Masked (LeRF)", fontsize=12)
    ax.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax.set_title("LeRF Curves by Masking Baseline\n(ResNet-50, Integrated Gradients)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_abpc_comparison(summary: dict, output_path: str):
    """Bar chart comparing MoRF and ABPC scores across baselines."""
    labels   = [BASELINE_LABELS[b] for b in BASELINES]
    morf_vals = [summary[b]["mean_morf"] for b in BASELINES]
    abpc_vals = [summary[b]["mean_abpc"] for b in BASELINES]
    colors    = [BASELINE_COLORS[b] for b in BASELINES]

    x    = np.arange(len(BASELINES))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    bars1 = ax.bar(x - w/2, morf_vals, w, label="MoRF Score ↑", color=colors, alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + w/2, abpc_vals, w, label="ABPC Score ↑",
                   color=colors, alpha=0.5, edgecolor='black', hatch='//')

    # Value labels on bars
    for bar in bars1:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=10)
    ax.set_ylabel("Faithfulness Score", fontsize=12)
    ax.set_title("Faithfulness Scores by Masking Baseline\n"
                 "ResNet-50 + Integrated Gradients (Phase 2: Baseline Sensitivity)",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.4)
    ax.set_ylim(0, max(morf_vals + abpc_vals) * 1.25 + 0.05)

    # Reference line: paper's grey baseline MoRF
    ax.axhline(y=summary["grey"]["mean_morf"], color="red",
               linestyle='--', linewidth=1.2, alpha=0.7,
               label=f"M4 Paper Baseline (grey MoRF={summary['grey']['mean_morf']:.3f})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_sensitivity_heatmap(all_morf: dict, all_abpc: dict, output_path: str):
    """
    Baseline Sensitivity Score (BSS): std of each score across baselines per image.
    Plot distribution of BSS.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Per-image MoRF as matrix: (n_images, n_baselines)
    n_images = len(all_morf[BASELINES[0]])
    morf_mat = np.array([[all_morf[b][i] for b in BASELINES] for i in range(n_images)])
    abpc_mat = np.array([[all_abpc[b][i] for b in BASELINES] for i in range(n_images)])

    bss_morf = morf_mat.std(axis=1)
    bss_abpc = abpc_mat.std(axis=1)

    ax = axes[0]
    ax.hist(bss_morf, bins=20, color='#E63946', alpha=0.8, edgecolor='black')
    ax.set_xlabel("BSS (std of MoRF across baselines)", fontsize=11)
    ax.set_ylabel("Image Count", fontsize=11)
    ax.set_title(f"Baseline Sensitivity Score — MoRF\n"
                 f"Mean BSS = {bss_morf.mean():.4f}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(bss_abpc, bins=20, color='#457B9D', alpha=0.8, edgecolor='black')
    ax.set_xlabel("BSS (std of ABPC across baselines)", fontsize=11)
    ax.set_ylabel("Image Count", fontsize=11)
    ax.set_title(f"Baseline Sensitivity Score — ABPC\n"
                 f"Mean BSS = {bss_abpc.mean():.4f}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 2: Vision Baseline Sensitivity Analysis"
    )
    p.add_argument("--data_dir",    default="data/M4_5000",
                   help="ImageFolder-style directory with synset subdirs")
    p.add_argument("--n_images",    type=int, default=100,
                   help="Number of images to evaluate (100 for speed)")
    p.add_argument("--n_steps",     type=int, default=N_STEPS,
                   help="Perturbation steps (paper: 20)")
    p.add_argument("--ig_steps",    type=int, default=50,
                   help="IG integration steps (paper: 50)")
    p.add_argument("--output_dir",  default="results/phase2")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Data dir    : {args.data_dir}")
    print(f"n_images    : {args.n_images}")
    print(f"n_steps     : {args.n_steps}")
    print(f"ig_steps    : {args.ig_steps}")
    print(f"Baselines   : {BASELINES}")

    # Load model
    print("\nLoading ResNet-50 (IMAGENET1K_V1)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)

    # Load records
    print(f"Loading dataset from {args.data_dir}...")
    try:
        ds = ImageFolder(args.data_dir)
        all_records = [(p, c) for p, c in ds.samples]
    except Exception as e:
        print(f"[error] Could not load dataset: {e}")
        return

    print(f"Found {len(all_records)} images total.")

    # Sample uniformly across classes
    rng_py = random.Random(args.seed)
    rng_py.shuffle(all_records)
    records = all_records[:args.n_images]
    print(f"Using {len(records)} images.")

    # Per-baseline accumulators
    all_morf_scores = {b: [] for b in BASELINES}
    all_abpc_scores = {b: [] for b in BASELINES}
    all_morf_curves = {b: [] for b in BASELINES}
    all_lerf_curves = {b: [] for b in BASELINES}

    print("\nRunning evaluation...")
    t0 = time.time()
    processed = 0

    for idx, (img_path, _) in enumerate(records):
        result = evaluate_image(model, img_path, device, ig_steps=args.ig_steps)
        if result is None:
            continue

        processed += 1
        for b in BASELINES:
            all_morf_scores[b].append(result[b]["morf_score"])
            all_abpc_scores[b].append(result[b]["abpc_score"])
            all_morf_curves[b].append(result[b]["morf_curve"])
            all_lerf_curves[b].append(result[b]["lerf_curve"])

        if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
            elapsed = time.time() - t0
            rate    = (idx + 1) / elapsed
            eta     = (len(records) - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{len(records)}]  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                  flush=True)

    if processed == 0:
        print("[error] No images processed successfully.")
        return

    print(f"\nProcessed {processed} images.")

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = {}
    print("\n" + "="*70)
    print("  Phase 2: Baseline Sensitivity Results — ResNet-50 + IG")
    print("="*70)
    print(f"{'Baseline':<22} {'MoRF Score (^)':>16} {'ABPC Score (^)':>16}")
    print("-"*56)

    for b in BASELINES:
        m_morf = float(np.mean(all_morf_scores[b]))
        m_abpc = float(np.mean(all_abpc_scores[b]))
        summary[b] = {"mean_morf": m_morf, "mean_abpc": m_abpc}
        flag = " <- M4 paper" if b == "grey" else ""
        print(f"  {BASELINE_LABELS[b]:<22} {m_morf:>14.4f}   {m_abpc:>14.4f}{flag}")

    print("="*70)

    # Baseline Sensitivity Score (BSS)
    morf_mat = np.array([[all_morf_scores[b][i] for b in BASELINES]
                          for i in range(processed)])
    abpc_mat = np.array([[all_abpc_scores[b][i] for b in BASELINES]
                          for i in range(processed)])
    bss_morf = float(morf_mat.std(axis=1).mean())
    bss_abpc = float(abpc_mat.std(axis=1).mean())
    print(f"\n  Baseline Sensitivity Score (BSS-MoRF): {bss_morf:.4f}")
    print(f"  Baseline Sensitivity Score (BSS-ABPC): {bss_abpc:.4f}")
    print(f"  (BSS = mean std of scores across baselines per image)")
    print(f"  Higher BSS → metrics are more sensitive to baseline choice")

    # ── Mean perturbation curves ───────────────────────────────────────────────
    mean_curves = {}
    for b in BASELINES:
        mean_curves[b] = {
            "morf": np.mean(all_morf_curves[b], axis=0),
            "lerf": np.mean(all_lerf_curves[b], axis=0),
        }

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_morf_curves(mean_curves, args.n_steps,
                     os.path.join(args.output_dir, "vision_morf_lerf_curves.png"))
    plot_abpc_comparison(summary,
                         os.path.join(args.output_dir, "vision_abpc_comparison.png"))
    plot_sensitivity_heatmap(all_morf_scores, all_abpc_scores,
                             os.path.join(args.output_dir, "vision_bss_distribution.png"))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "vision_baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["baseline", "MoRF", "ABPC", "N", "BSS_MoRF", "BSS_ABPC",
                    "ig_steps", "n_steps", "seed"])
        for b in BASELINES:
            w.writerow([
                b,
                f"{summary[b]['mean_morf']:.4f}",
                f"{summary[b]['mean_abpc']:.4f}",
                processed,
                f"{bss_morf:.4f}",
                f"{bss_abpc:.4f}",
                args.ig_steps,
                args.n_steps,
                args.seed,
            ])
    print(f"  Saved: {csv_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()


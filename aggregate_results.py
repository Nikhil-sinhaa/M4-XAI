"""
Collect the per-sample results into the paper's tables and figures.

    python aggregate_results.py

Writes to results/m4/summary/:
    vision_scores.csv / text_scores.csv   mean MoRF, LeRF, ABPC, PScore, INFD, SynScore per model x method
    table1_resnet50.md                    Table 1 (ours vs. paper)
    avgscore_vision.png / _text.png       Fig. 3: AvgScore = z(MoRF)+z(ABPC)+z(PScore)-z(INFD)+z(SynScore)
    metric_correlation.png                Fig. 2: Pearson correlation of the metrics over model x method pairs
    sensitivity_vision.png / _text.png    Figs. 4-5: std of each metric across methods, per model
    avgscore_*.csv, sensitivity_*.csv     the numbers behind every figure (table view)
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

VISION_ORDER = ["mobilenet", "vgg16", "resnet50", "resnet101", "resnet152",
                "vit_small", "vit_base", "vit_large", "mae_vit_base"]
VISION_LABEL = {"mobilenet": "MobileNet", "vgg16": "VGG16", "resnet50": "R50", "resnet101": "R101",
                "resnet152": "R152", "vit_small": "ViT/S", "vit_base": "ViT/B", "vit_large": "ViT/L",
                "mae_vit_base": "ViT/MAE"}
TEXT_ORDER = ["distilbert", "bert_base", "bert_large", "ernie_2_base", "roberta_base"]
TEXT_LABEL = {"distilbert": "DistilBert", "bert_base": "Bert/B", "bert_large": "Bert/L",
              "ernie_2_base": "Ernie-2.0", "roberta_base": "Roberta/B"}
METHOD_ORDER = ["bt_head", "bt_token", "ga", "ig", "sg", "gradcam", "lime"]
METHOD_LABEL = {"bt_head": "BT/H", "bt_token": "BT/T", "ga": "GA", "ig": "IG", "sg": "SG",
                "gradcam": "GradCam", "lime": "LIME", "constant": "Constant", "random_16": "Random-16",
                "random": "Random"}
BASELINES = {"constant", "random_16", "random"}
METRICS = ["MoRF", "ABPC", "PScore", "INFD", "SynScore"]

PAPER_TABLE1 = {   # NeurIPS 2023, Table 1 (ResNet-50): MoRF, ABPC, PScore, INFD, SynScore
    "constant": (0.000, 0.000, None, 3.015, 0.072),
    "random_16": (0.596, 0.007, None, 3.039, 0.077),
    "random": (0.599, 0.008, None, 3.015, 0.078),
    "gradcam": (0.628, 0.424, 0.835, 2.496, 1.000),
    "ig": (0.709, 0.377, 0.812, 2.373, 0.999),
    "sg": (0.701, 0.369, 0.820, 2.323, 0.998),
}

# Reference palette (dataviz skill): categorical slots 1-5, blue<->red diverging with a gray midpoint.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("m4div", ["#b8302f", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])


def mean_col(path: Path, col: str):
    if not path.exists():
        return None, 0
    with open(path) as f:
        vals = [float(r[col]) for r in csv.DictReader(f) if r[col] not in ("", "nan")]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def collect(root: Path, order):
    rows = []
    for model in order:
        for d in sorted((root / model).glob("*/")) if (root / model).exists() else []:
            if not (d / "scores.csv").exists() and not (d / "synscore.csv").exists():
                continue
            r = {"model": model, "method": d.name}
            for k in ("MoRF", "LeRF", "ABPC", "INFD"):
                r[k], r["N"] = mean_col(d / "scores.csv", k)
            r["PScore"], _ = mean_col(d / "pscore.csv", "PScore")
            r["SynScore"], r["N_syn"] = mean_col(d / "synscore.csv", "SynScore_AP")
            rows.append(r)
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]))
                        for k in fields})


def fmt(v):
    return "N/A" if v is None else f"{v:.3f}"


def pick_layer_variants(rows, rule: str):
    """Collapse 'ga-4', 'bt_head-8', ... to one row per (model, method) following `rule`."""
    plain = [r for r in rows if "-" not in r["method"]]
    groups = {}
    for r in rows:
        if "-" in r["method"]:
            base, sl = r["method"].rsplit("-", 1)
            groups.setdefault((r["model"], base), []).append((int(sl), r))
    for (model, base), cands in groups.items():
        if any(r["model"] == model and r["method"] == base for r in plain):
            continue                                   # an un-suffixed run of this method exists: keep it
        if rule == "best_abpc":
            sl, r = max(cands, key=lambda c: -np.inf if c[1]["ABPC"] is None else c[1]["ABPC"])
        elif rule == "last":
            sl, r = max(cands, key=lambda c: c[0])
        else:
            match = [c for c in cands if c[0] == int(rule)]
            if not match:
                continue
            sl, r = match[0]
        plain.append({**r, "method": base, "start_layer": sl})
        print(f"  {model}/{base}: start layer {sl} ({rule})")
    return plain


def table1(rows, out: Path):
    ours = {r["method"]: r for r in rows if r["model"] == "resnet50"}
    lines = ["# Table 1 — ResNet-50 (ours vs. paper)", "",
             "| Method | MoRF↑ ours / paper | ABPC↑ | PScore↑ | INFD↓ | SynScore↑ | N |",
             "|---|---|---|---|---|---|---|"]
    for m, ref in PAPER_TABLE1.items():
        r = ours.get(m, {})
        cells = [f"{fmt(r.get(k))} / {fmt(p)}" for k, p in zip(METRICS, ref)]
        lines.append(f"| {METHOD_LABEL[m]} | " + " | ".join(cells) + f" | {r.get('N', 0)} |")
    out.write_text("\n".join(lines) + "\n")


def avgscore(rows, order, labels, methods_allowed, title, out_png: Path, out_csv: Path):
    """Eq. (7): z-standardise each metric across the methods of one model, INFD negated, then sum."""
    grid = {}
    for model in order:
        rs = [r for r in rows if r["model"] == model and r["method"] in methods_allowed]
        if len(rs) < 2:
            continue
        total = np.zeros(len(rs))
        for k in METRICS:
            v = np.array([np.nan if r.get(k) is None else r[k] for r in rs], float)
            if np.isnan(v).any() or v.std() == 0:
                continue                      # metric missing for some method of this model: leave it out
            z = (v - v.mean()) / v.std()
            total += -z if k == "INFD" else z
        for r, s in zip(rs, total):
            grid[(model, r["method"])] = s
    models = [m for m in order if any(k[0] == m for k in grid)]
    methods = [m for m in METHOD_ORDER if any(k[1] == m for k in grid)]
    if not models:
        return
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + methods)
        for m in models:
            w.writerow([m] + [f"{grid[(m, k)]:.3f}" if (m, k) in grid else "" for k in methods])

    A = np.array([[grid.get((m, k), np.nan) for k in methods] for m in models])
    lim = max(1e-6, np.nanmax(np.abs(A)))
    fig, ax = plt.subplots(figsize=(1.0 * len(methods) + 2, 0.55 * len(models) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(A), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(models)):
        for j in range(len(methods)):
            if not np.isnan(A[i, j]):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if abs(A[i, j]) > 0.6 * lim else INK)
    ax.set_xticks(range(len(methods)), [METHOD_LABEL[m] for m in methods], color=INK2)
    ax.set_yticks(range(len(models)), [labels[m] for m in models], color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    cb.set_label("AvgScore (higher = more faithful); blank = method not applicable", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def correlation(rows, out_png: Path, out_csv: Path):
    keys = [k for k in METRICS if sum(r.get(k) is not None for r in rows) >= 3]
    rs = [r for r in rows if all(r.get(k) is not None for k in keys)]
    if len(rs) < 3 or len(keys) < 2:
        return
    X = np.array([[r[k] for k in keys] for r in rs])
    C = np.corrcoef(X.T)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + keys)
        for k, row in zip(keys, C):
            w.writerow([k] + [f"{c:.3f}" for c in row])
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    im = ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if abs(C[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(keys)), keys, color=INK2)
    ax.set_yticks(range(len(keys)), keys, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Metric correlation over {len(rs)} model × method pairs", color=INK, loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def sensitivity(rows, order, labels, methods_allowed, metrics, title, out_png: Path, out_csv: Path):
    """Figs. 4-5: standard deviation of each metric across the attribution methods of a model."""
    data = {}
    for model in order:
        rs = [r for r in rows if r["model"] == model and r["method"] in methods_allowed]
        for k in metrics:
            v = [r[k] for r in rs if r.get(k) is not None]
            if len(v) >= 2:
                data[(model, k)] = float(np.std(v))
    models = [m for m in order if any(key[0] == m for key in data)]
    ks = [k for k in metrics if any(key[1] == k for key in data)]
    if not models:
        return
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + ks)
        for m in models:
            w.writerow([m] + [f"{data[(m, k)]:.4f}" if (m, k) in data else "" for k in ks])
    fig, ax = plt.subplots(figsize=(1.1 * len(models) + 2, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.8 / len(ks)
    for j, k in enumerate(ks):
        xs = [i + (j - (len(ks) - 1) / 2) * width for i in range(len(models))]
        ax.bar(xs, [data.get((m, k), 0) for m in models], width * 0.85, color=SERIES[j], label=k)
    ax.set_xticks(range(len(models)), [labels[m] for m in models], color=INK2)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_ylabel("std across attribution methods", color=MUTED)
    ax.legend(frameon=False, ncol=len(ks), loc="upper left", labelcolor=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/m4")
    p.add_argument("--layer_rule", default="best_abpc",
                   help="which GA/BT start-layer variant enters Fig. 3/5 when several were run: 'best_abpc' "
                        "(highest ABPC per model), 'last' (deepest start layer) or an integer layer. The paper "
                        "does not state its choice; every variant is listed in the *_scores.csv tables.")
    a = p.parse_args()
    root = Path(a.root)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    fields = ["model", "method", "MoRF", "LeRF", "ABPC", "PScore", "INFD", "SynScore", "N", "N_syn"]

    vrows = collect(root / "vision", VISION_ORDER)
    if vrows:
        write_csv(out / "vision_scores.csv", vrows, fields)
        table1(vrows, out / "table1_resnet50.md")
        vrows = pick_layer_variants(vrows, a.layer_rule)
        vm = {"bt_head", "bt_token", "ga", "ig", "sg", "gradcam"}
        avgscore(vrows, VISION_ORDER, VISION_LABEL, vm, "AvgScore — images (Fig. 3a)",
                 out / "avgscore_vision.png", out / "avgscore_vision.csv")
        correlation([r for r in vrows if r["method"] not in BASELINES],
                    out / "metric_correlation.png", out / "metric_correlation.csv")
        sensitivity(vrows, VISION_ORDER, VISION_LABEL, vm, METRICS, "Sensitivity to attribution method (Fig. 4)",
                    out / "sensitivity_vision.png", out / "sensitivity_vision.csv")

    trows = collect(root / "text", TEXT_ORDER)
    if trows:
        write_csv(out / "text_scores.csv", trows, fields)
        canon = pick_layer_variants(trows, a.layer_rule)
        avgscore(canon, TEXT_ORDER, TEXT_LABEL, set(METHOD_ORDER), "AvgScore — texts (Fig. 3b)",
                 out / "avgscore_text.png", out / "avgscore_text.csv")
        sensitivity(canon, TEXT_ORDER, TEXT_LABEL, set(METHOD_ORDER), ["MoRF", "ABPC", "PScore"],
                    "Sensitivity to attribution method (Fig. 5)", out / "sensitivity_text.png",
                    out / "sensitivity_text.csv")
    print(f"summary written to {out}/ ({len(vrows)} vision rows, {len(trows)} text rows)")


if __name__ == "__main__":
    main()

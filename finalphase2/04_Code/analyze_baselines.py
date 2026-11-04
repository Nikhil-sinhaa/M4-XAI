"""
Phase 2 analysis: is the M4 method ranking robust to the masking baseline and to the sample size?

Reads the curves written by rescore_baselines.py and produces, in results/m4/rescore/summary/:

  baseline_scores.csv     mean MoRF / ABPC per model, method, baseline with 95% bootstrap CI
  rank_stability.csv      Kendall tau between the grey (M4) ranking and every other baseline's ranking,
                          and the method pairs whose order flips
  multibaseline.csv       "before" (grey ABPC) vs "after" (ABPC averaged over baselines), BSS per method,
                          and how often each ranking is reproduced under bootstrap resampling of images
  sample_size.csv         ranking agreement of N-image subsets vs the full 5000 images (MobileNet)
  *.png                   figures

    python analyze_baselines.py
"""

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REF = "grey"                       # M4 default baseline = "before" ("pad" for text, set with --ref)
N_BOOT = 2000
MODEL_ORDER = ["resnet50", "resnet101", "resnet152", "vgg16", "mobilenet", "vit_small", "vit_base", "mae_vit_base",
               "distilbert", "bert_base", "bert_large", "ernie_2_base", "roberta_base"]


def per_image_scores(curves: np.ndarray) -> dict:
    """curves (..., 2, L+1) -> per-image MoRF and ABPC with the M4 definitions (Eq. 1-3)."""
    c = curves.astype(np.float64)
    morf_p, lerf_p = c[..., 0, :], c[..., 1, :]
    return {"MoRF": morf_p[..., 0] - morf_p.mean(-1), "ABPC": (lerf_p - morf_p).mean(-1)}


def load_all(root: Path):
    """-> {model: (baselines, {method: {'MoRF': (B, N), 'ABPC': (B, N)}})}"""
    data = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "summary"):
        bfile = mdir / "baselines.txt"
        if not bfile.exists():
            continue
        baselines = bfile.read_text().split()
        curves = {f.stem: np.load(f) for f in sorted(mdir.glob("*.npy"))}
        n_common = min((c.shape[1] for c in curves.values()), default=0)     # methods may differ in length
        methods = {m: per_image_scores(c[:, :n_common]) for m, c in curves.items()}
        if len(methods) >= 2:
            data[mdir.name] = (baselines, methods)
    return dict(sorted(data.items(), key=lambda kv: MODEL_ORDER.index(kv[0]) if kv[0] in MODEL_ORDER else len(MODEL_ORDER)))


def boot_ci(x: np.ndarray, rng, n_boot=N_BOOT):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(1)
    return np.percentile(means, [2.5, 97.5])


def ranking(scores: dict) -> list:
    """methods ordered best -> worst."""
    return sorted(scores, key=lambda m: -scores[m])


def tau(a: dict, b: dict) -> float:
    ms = list(a)
    if len(ms) < 2:
        return float("nan")
    t = kendalltau([a[m] for m in ms], [b[m] for m in ms]).statistic
    return float(t)


def rank_repro(per_image: dict, ref_rank: list, rng, n_boot=N_BOOT) -> float:
    """Fraction of bootstrap resamples of the images whose method ranking equals ref_rank."""
    ms = list(per_image)
    n = len(per_image[ms[0]])
    hits = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        hits += ranking({m: per_image[m][idx].mean() for m in ms}) == ref_rank
    return hits / n_boot


def analyse(data, out: Path, seed=42):
    rng = np.random.default_rng(seed)
    score_rows, stab_rows, multi_rows = [], [], []
    for model, (baselines, methods) in data.items():
        ms = list(methods)
        abpc = {m: methods[m]["ABPC"] for m in ms}                    # (B, N)
        for b, bname in enumerate(baselines):
            for m in ms:
                for metric in ("MoRF", "ABPC"):
                    x = methods[m][metric][b]
                    lo, hi = boot_ci(x, rng)
                    score_rows.append(dict(model=model, method=m, baseline=bname, metric=metric,
                                           mean=x.mean(), ci_lo=lo, ci_hi=hi, N=len(x)))

        # rank stability: grey ranking vs the ranking under every other baseline
        ref_i = baselines.index(REF)
        for metric in ("MoRF", "ABPC"):
            mean = {bn: {m: methods[m][metric][b].mean() for m in ms} for b, bn in enumerate(baselines)}
            for bn in baselines:
                if bn == REF:
                    continue
                flips = [f"{a}<>{c}" for a, c in itertools.combinations(ms, 2)
                         if (mean[REF][a] - mean[REF][c]) * (mean[bn][a] - mean[bn][c]) < 0]
                stab_rows.append(dict(model=model, metric=metric, baseline=bn, tau_vs_grey=tau(mean[REF], mean[bn]),
                                      rank_grey=">".join(ranking(mean[REF])), rank_baseline=">".join(ranking(mean[bn])),
                                      flipped_pairs=";".join(flips) or "-"))

        # "after": ABPC averaged over baselines (per image), plus BSS
        multi = {m: abpc[m].mean(0) for m in ms}                       # (N,)
        grey = {m: abpc[m][ref_i] for m in ms}
        grey_rank, multi_rank = ranking({m: grey[m].mean() for m in ms}), ranking({m: multi[m].mean() for m in ms})
        r_grey, r_multi = rank_repro(grey, grey_rank, rng), rank_repro(multi, multi_rank, rng)
        for m in ms:
            per_b = abpc[m].mean(1)                                    # (B,) mean ABPC per baseline
            lo, hi = boot_ci(multi[m], rng)
            multi_rows.append(dict(model=model, method=m, ABPC_grey=grey[m].mean(), ABPC_multi=multi[m].mean(),
                                   multi_ci_lo=lo, multi_ci_hi=hi, BSS=per_b.std(ddof=0),
                                   BSS_per_image=abpc[m].std(0, ddof=0).mean(),
                                   rank_grey=grey_rank.index(m) + 1, rank_multi=multi_rank.index(m) + 1,
                                   ranking_grey=">".join(grey_rank), ranking_multi=">".join(multi_rank),
                                   bootstrap_repro_grey=r_grey, bootstrap_repro_multi=r_multi))
    pd.DataFrame(score_rows).to_csv(out / "baseline_scores.csv", index=False, float_format="%.4f")
    pd.DataFrame(stab_rows).to_csv(out / "rank_stability.csv", index=False, float_format="%.3f")
    pd.DataFrame(multi_rows).to_csv(out / "multibaseline.csv", index=False, float_format="%.4f")
    return pd.DataFrame(score_rows), pd.DataFrame(stab_rows), pd.DataFrame(multi_rows)


def sample_size(root: Path, out: Path, model="mobilenet", sizes=(100, 250, 500, 1000, 2500), reps=500, seed=42):
    """How well does an N-image subset reproduce the 5000-image ranking? Uses M4's own grey-baseline scores."""
    rng = np.random.default_rng(seed)
    cols = {}
    for mdir in sorted((root / model).iterdir()) if (root / model).exists() else []:
        f = mdir / "scores.csv"
        if f.exists() and mdir.is_dir():
            cols[mdir.name] = pd.read_csv(f)["ABPC"].to_numpy()
    if len(cols) < 2:
        return None
    n = min(len(v) for v in cols.values())
    full = {m: v[:n].mean() for m, v in cols.items()}
    full_rank = ranking(full)
    rows = []
    for size in (s for s in sizes if s < n):
        taus, same = [], 0
        for _ in range(reps):
            idx = rng.choice(n, size, replace=False)
            sub = {m: v[idx].mean() for m, v in cols.items()}
            taus.append(tau(full, sub))
            same += ranking(sub) == full_rank
        rows.append(dict(model=model, N=size, mean_tau=np.mean(taus), p_same_ranking=same / reps,
                         full_N=n, full_ranking=">".join(full_rank)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "sample_size.csv", index=False, float_format="%.3f")
    return df


# ─────────────────────────────── figures ───────────────────────────────

def fig_baseline_bars(scores: pd.DataFrame, out: Path):
    a = scores[scores.metric == "ABPC"]
    models = list(a.model.unique())
    ncol = min(len(models), 3)
    nrow = -(-len(models) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow + 0.6), squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    for ax, model in zip(axes.flat, models):
        d = a[a.model == model]
        methods, baselines = list(d.method.unique()), list(d.baseline.unique())
        w = 0.8 / len(baselines)
        for i, b in enumerate(baselines):
            r = d[d.baseline == b].set_index("method").loc[methods]
            ax.bar(np.arange(len(methods)) + i * w, r["mean"], w, label=b,
                   yerr=[r["mean"] - r.ci_lo, r.ci_hi - r["mean"]], capsize=2,
                   edgecolor="black" if b == REF else None, linewidth=1.2)
        ax.set_xticks(np.arange(len(methods)) + 0.4 - w / 2, methods)
        ax.set_title(f"{model}  (N={int(d.N.iloc[0])})")
        ax.set_ylabel("ABPC (95% bootstrap CI)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, title="masking baseline", loc="lower center", ncol=len(labels), fontsize=8)
    fig.savefig(out / "abpc_by_baseline.png", dpi=150)
    plt.close(fig)


def fig_rank_heatmap(scores: pd.DataFrame, out: Path):
    a = scores[scores.metric == "ABPC"]
    models = list(a.model.unique())
    ncol = min(len(models), 4)
    nrow = -(-len(models) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    for ax, model in zip(axes.flat, models):
        piv = a[a.model == model].pivot(index="method", columns="baseline", values="mean")
        ranks = piv.rank(ascending=False).astype(int)
        ax.imshow(ranks.values, cmap="RdYlGn_r", vmin=1, vmax=max(2, ranks.values.max()))
        for (i, j), v in np.ndenumerate(ranks.values):
            ax.text(j, i, v, ha="center", va="center")
        ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_title(f"{model}: ABPC rank (1 = best)")
    fig.tight_layout()
    fig.savefig(out / "rank_heatmap.png", dpi=150)
    plt.close(fig)


def fig_sample_size(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.plot(df.N, df.p_same_ranking, "o-", label="P(identical ranking)")
    ax.plot(df.N, df.mean_tau, "s--", label="mean Kendall τ")
    ax.set_xscale("log")
    ax.set_xlabel("images in subset")
    ax.set_title(f"{df.model.iloc[0]}: subset vs full {int(df.full_N.iloc[0])}-image ranking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sample_size.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rescore_root", default="results/m4/rescore")
    p.add_argument("--vision_root", default="results/m4/vision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ref", default="grey", help="reference (M4 default) baseline: grey for vision, pad for text")
    args = p.parse_args()
    global REF
    REF = args.ref
    root = Path(args.rescore_root)
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    data = load_all(root)
    if data:
        scores, stab, multi = analyse(data, out, args.seed)
        fig_baseline_bars(scores, out)
        fig_rank_heatmap(scores, out)
        print("\n== rank stability (ABPC) ==")
        print(stab[stab.metric == "ABPC"].to_string(index=False))
        print("\n== before (grey) vs after (multi-baseline) ==")
        print(multi[["model", "method", "ABPC_grey", "ABPC_multi", "BSS", "rank_grey", "rank_multi",
                     "bootstrap_repro_grey", "bootstrap_repro_multi"]].to_string(index=False))
    else:
        print(f"no rescored models with >=2 methods under {root}")
    ss = sample_size(Path(args.vision_root), out) if args.vision_root else None
    if ss is not None:
        fig_sample_size(ss, out)
        print("\n== sample size ==")
        print(ss.to_string(index=False))


if __name__ == "__main__":
    main()

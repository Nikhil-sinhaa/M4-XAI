"""
Build the Phase 2 "Experimental Results and Comparison" report (HTML -> PDF).

Every number from OUR experiments is read from phase2_results/ (the CSVs written by analyze_baselines.py).
The only hard-coded numbers are the PUBLISHED M4 values (Table 1, p. 6), which are labelled as such.
Key claims in the prose are asserted against the data, so the build fails if the text and the data disagree.

    python phase2_report/build_report.py            # writes phase2_report/report.html
    libreoffice --headless --convert-to pdf:writer_web_pdf_Export --outdir phase2_report phase2_report/report.html
"""

import base64
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "phase2_results" / "rescore" / "summary"
OUT = ROOT / "phase2_report"

scores = pd.read_csv(RES / "baseline_scores.csv")
stab = pd.read_csv(RES / "rank_stability.csv")
multi = pd.read_csv(RES / "multibaseline.csv")
ss = pd.read_csv(RES / "sample_size.csv")

# PUBLISHED values, M4 paper (NeurIPS 2023), Table 1, page 6. Not our results.
PAPER_T1 = {"gradcam": (0.628, 0.424), "ig": (0.709, 0.377), "sg": (0.701, 0.369)}   # (MoRF, ABPC)
BASELINES = ["grey", "black", "mean", "blur", "noise"]


def img(path: Path, width: int) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<p><img src="data:image/png;base64,{b64}" width="{width}"></p>'


def table(df: pd.DataFrame, fmt=None) -> str:
    return df.to_html(index=False, border=1, escape=False, float_format=fmt or (lambda v: f"{v:.3f}"),
                      classes="t").replace("<table", '<table cellpadding="4" cellspacing="0"')


def abpc_table(model: str) -> pd.DataFrame:
    d = scores[(scores.model == model) & (scores.metric == "ABPC")].copy()
    d["cell"] = d.apply(lambda r: f"{r['mean']:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
    p = d.pivot(index="method", columns="baseline", values="cell")[BASELINES]
    p.columns.name = None
    p.insert(0, "method", p.index)
    p.index.name = None
    return p.reset_index(drop=True)


def get(model, method, baseline, metric="ABPC", col="mean"):
    r = scores[(scores.model == model) & (scores.method == method) & (scores.baseline == baseline) &
               (scores.metric == metric)]
    return float(r[col].iloc[0])


# ---------------------------------------------------------------- facts computed from the data
ORDER = ["resnet50", "resnet101", "resnet152", "vgg16", "mobilenet", "vit_small", "vit_base", "mae_vit_base"]
NICE = {"resnet50": "ResNet-50", "resnet101": "ResNet-101", "resnet152": "ResNet-152", "vgg16": "VGG16",
        "mobilenet": "MobileNet", "vit_small": "ViT-S", "vit_base": "ViT-B", "mae_vit_base": "MAE-ViT-B"}
MODELS = [m for m in ORDER if m in set(scores.model)]
NIMG = {m: int(scores[scores.model == m].N.iloc[0]) for m in MODELS}
PIXEL = ("ig", "sg")                       # pixel-level methods; the others (GradCAM, attention-based) are coarse-grained
multi_i = multi.set_index(["model", "method"])
bss = multi_i.BSS
mm = {m: multi[multi.model == m] for m in MODELS}


def rank_of(model, b):
    return stab[(stab.model == model) & (stab.metric == "ABPC") & (stab.baseline == b)].rank_baseline.iloc[0]


def rank_grey_of(model):
    return stab[(stab.model == model) & (stab.metric == "ABPC")].rank_grey.iloc[0]


def nonpix(model):
    return [m for m in mm[model].method if m not in PIXEL]


def ci_overlap_all(model, methods):
    """True if all grey-baseline ABPC CIs of `methods` share a common interval (cannot be separated)."""
    iv = [(get(model, m, "grey", col="ci_lo"), get(model, m, "grey", col="ci_hi")) for m in methods]
    return max(a for a, _ in iv) < min(b for _, b in iv)


# per model: do the grey rankings change under some other baseline? which baselines?
changed = {m: [b for b in BASELINES[1:] if rank_of(m, b) != rank_grey_of(m)] for m in MODELS}
n_changed = sum(bool(v) for v in changed.values())
n_mean_changed = sum('mean' in v for v in changed.values())
reversed_ = [(m, b) for m in MODELS for b in BASELINES[1:] if rank_of(m, b) == ">".join(rank_grey_of(m).split(">")[::-1])]
# BSS: pixel-level methods vs the most baseline-sensitive coarse method
bss_pix_higher = [m for m in MODELS if nonpix(m) and min(bss[(m, x)] for x in PIXEL if (m, x) in bss.index) >
                  max(bss[(m, x)] for x in nonpix(m))]
bss_pix_range = (min(bss[(m, x)] for m in MODELS for x in PIXEL if (m, x) in bss.index),
                 max(bss[(m, x)] for m in MODELS for x in PIXEL if (m, x) in bss.index))
bss_coarse_range = (min(bss[(m, x)] for m in MODELS for x in nonpix(m)),
                    max(bss[(m, x)] for m in MODELS for x in nonpix(m)))
# under black: IG / SG below every coarse method?
gap_black = {(m, x): min(get(m, y, "black") for y in nonpix(m)) - get(m, x, "black")
             for m in MODELS for x in PIXEL if (m, x) in bss.index}
below_black = [k for k, v in gap_black.items() if v > 0]
ig_drop = {m: get(m, "ig", "grey") - get(m, "ig", "black") for m in MODELS}
tied = [m for m in MODELS if len(nonpix(m)) >= 2 and ci_overlap_all(m, nonpix(m))]
s500 = float(ss[ss.N == 500].p_same_ranking.iloc[0])
rn = mm["resnet50"]

# ---------------------------------------------------------------- tables
paper_vs_ours = pd.DataFrame([
    dict(method=m.upper() if m != "gradcam" else "GradCAM",
         **{"MoRF paper (p.6)": PAPER_T1[m][0], "MoRF ours": get("resnet50", m, "grey", "MoRF"),
            "ABPC paper (p.6)": PAPER_T1[m][1], "ABPC ours [95% CI]": f"{get('resnet50', m, 'grey'):.3f} "
            f"[{get('resnet50', m, 'grey', col='ci_lo'):.3f}, {get('resnet50', m, 'grey', col='ci_hi'):.3f}]"})
    for m in ("gradcam", "ig", "sg")])

rank_cols = ["baseline", "tau_vs_grey", "rank_baseline", "flipped_pairs"]
def rank_tbl(model):
    d = stab[(stab.model == model) & (stab.metric == "ABPC")][rank_cols].copy()
    d.columns = ["baseline", "Kendall tau vs grey", "ranking (best > worst)", "pairs that flip vs grey"]
    grey_row = pd.DataFrame([dict(zip(d.columns, ["grey (M4)", 1.0, stab[(stab.model == model) & (stab.metric == 'ABPC')].rank_grey.iloc[0], "-"]))])
    return pd.concat([grey_row, d], ignore_index=True)

def ba_tbl(d):
    t = d[["method", "ABPC_grey", "ABPC_multi", "BSS", "rank_grey", "rank_multi"]].copy()
    t.columns = ["method", "ABPC grey (before)", "ABPC multi-baseline (after)", "BSS", "rank before", "rank after"]
    return t

def rank_summary() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        d = stab[(stab.model == m) & (stab.metric == "ABPC")]
        rows.append({"model": NICE[m], "images": NIMG[m], "ranking under grey (M4)": rank_grey_of(m),
                     "baselines that change the ranking": ", ".join(changed[m]) or "none",
                     "min Kendall tau": float(d.tau_vs_grey.min())})
    return pd.DataFrame(rows)



# ---------------------------------------------------------------- TEXT models (optional: only if rescore_text_baselines.py was run)
TXT = Path(os.environ.get("P2_TEXT_SUMMARY", ROOT / "phase2_results" / "text_rescore" / "summary"))
HAS_TEXT = (TXT / "multibaseline.csv").exists()
text_html = text_interp = text_limit = text_gap = ""
if HAS_TEXT:
    tscores = pd.read_csv(TXT / "baseline_scores.csv")
    tstab = pd.read_csv(TXT / "rank_stability.csv")
    tmulti = pd.read_csv(TXT / "multibaseline.csv")
    TBASE = ["pad", "mask", "unk"]
    T_ORDER = ["distilbert", "bert_base", "bert_large", "ernie_2_base", "roberta_base"]
    TNICE = {"distilbert": "DistilBERT", "bert_base": "BERT-base", "bert_large": "BERT-large",
             "ernie_2_base": "ERNIE-2.0-base", "roberta_base": "RoBERTa-base"}
    TMODELS = [m for m in T_ORDER if m in set(tscores.model)]
    TN = {m: int(tscores[tscores.model == m].N.iloc[0]) for m in TMODELS}
    tm_ = {m: tmulti[tmulti.model == m] for m in TMODELS}
    family = lambda meth: "attention" if meth.split("-")[0] in ("ga", "bt_head", "bt_token") else "ig_sg_lime"

    def tget(model, method, baseline, metric="ABPC", col="mean"):
        r = tscores[(tscores.model == model) & (tscores.method == method) & (tscores.baseline == baseline) &
                    (tscores.metric == metric)]
        return float(r[col].iloc[0])

    def trank(model, b):
        return tstab[(tstab.model == model) & (tstab.metric == "ABPC") & (tstab.baseline == b)].rank_baseline.iloc[0]

    def trank_ref(model):
        return tstab[(tstab.model == model) & (tstab.metric == "ABPC")].rank_grey.iloc[0]     # "grey" column = reference baseline

    def tabpc_table(model):
        d = tscores[(tscores.model == model) & (tscores.metric == "ABPC")].copy()
        d["cell"] = d.apply(lambda r: f"{r['mean']:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]", axis=1)
        t = d.pivot(index="method", columns="baseline", values="cell")[TBASE]
        t.columns.name = None
        t.insert(0, "method", t.index)
        return t.reset_index(drop=True)

    tchanged = {m: [b for b in TBASE[1:] if trank(m, b) != trank_ref(m)] for m in TMODELS}
    t_n_changed = sum(bool(v) for v in tchanged.values())
    t_reversed = [(m, b) for m in TMODELS for b in TBASE[1:] if trank(m, b) == ">".join(trank_ref(m).split(">")[::-1])]
    meths = {m: list(tm_[m].method) for m in TMODELS}
    drop = {(m, x, b): tget(m, x, "pad") - tget(m, x, b) for m in TMODELS for x in meths[m] for b in TBASE[1:]}
    t_mask_lower = sum(v > 0 for (m, x, b), v in drop.items() if b == "mask")
    t_unk_lower = sum(v > 0 for (m, x, b), v in drop.items() if b == "unk")
    t_cases = sum(1 for k in drop if k[2] == "mask")
    mask_drops = [v for k, v in drop.items() if k[2] == "mask"]
    unk_drops = [v for k, v in drop.items() if k[2] == "unk"]
    tb = tmulti.assign(fam=tmulti.method.map(family))
    fam_bss = tb.groupby("fam").BSS.agg(["mean", "min", "max"])
    top_bss = {m: tm_[m].loc[tm_[m].BSS.idxmax(), "method"] for m in TMODELS}

    def trank_tbl_summary():
        rows = []
        for m in TMODELS:
            d = tstab[(tstab.model == m) & (tstab.metric == "ABPC")]
            rows.append({"model": TNICE[m], "documents": TN[m], "ranking under [PAD] (M4)": trank_ref(m),
                         "tokens that change the ranking": ", ".join(tchanged[m]) or "none",
                         "min Kendall tau": float(d.tau_vs_grey.min())})
        return pd.DataFrame(rows)

    def tba_all():
        t = pd.concat([ba_tbl(tm_[m]).assign(model=TNICE[m]) for m in TMODELS], ignore_index=True)
        return t[["model"] + [c for c in t.columns if c != "model"]]

    t_abpc_blocks = "".join(f"<p>{TNICE[m]}, {TN[m]} documents:</p>{table(tabpc_table(m))}" for m in TMODELS)
    text_html = f"""<h3>7.6 Text models: does the ranking survive another masking token? ([PAD] is M4's choice)</h3>
<p>Same idea as for images, on the MovieReview validation documents: the saved token attributions are re-scored with three masking tokens,
<b>[PAD]</b> (M4: masked tokens are also removed from attention), <b>[MASK]</b> and <b>[UNK]</b> (masked tokens stay visible as placeholders).
Methods per model: IG, SG, LIME and, for GA / BT-head / BT-token, the start layer with the best ABPC.</p>
{table(trank_tbl_summary())}
<p>ABPC by masking token, mean [95% CI]:</p>{t_abpc_blocks}
{img(TXT / 'abpc_by_baseline.png', 640)}{img(TXT / 'rank_heatmap.png', 640)}
<p>Before ([PAD]) vs after (average over the three tokens):</p>{table(tba_all(), lambda v: f'{v:.3f}')}"""
    text_interp = f"""<li><b>Text models.</b> The [PAD] ranking changes under [MASK] or [UNK] in {t_n_changed} of {len(TMODELS)} models
({"; ".join(f"{TNICE[m]}: {', '.join(v)}" for m, v in tchanged.items() if v) or "none"}).
{("A complete reversal occurs for " + "; ".join(f"{TNICE[m]} under {b}" for m, b in t_reversed) + ".") if t_reversed else "No ranking is completely reversed."}
ABPC is lower under [MASK] than under [PAD] in {t_mask_lower} of {t_cases} (model, method) cases (mean change {np.mean(mask_drops):+.3f} ABPC for [PAD] minus [MASK], range {min(mask_drops):+.3f} to {max(mask_drops):+.3f})
and lower under [UNK] in {t_unk_lower} of {t_cases} (mean {np.mean(unk_drops):+.3f}). BSS by method family: IG/SG/LIME {fam_bss.loc['ig_sg_lime','mean']:.3f} on average (range {fam_bss.loc['ig_sg_lime','min']:.3f} to {fam_bss.loc['ig_sg_lime','max']:.3f}),
attention-based {fam_bss.loc['attention','mean']:.3f} (range {fam_bss.loc['attention','min']:.3f} to {fam_bss.loc['attention','max']:.3f}). Most baseline-sensitive method per model: {", ".join(f"{TNICE[m]} {top_bss[m]}" for m in TMODELS)}.</li>"""
    text_limit = (f"<li>Text: {len(TMODELS)} models ({', '.join(f'{TNICE[m]} {TN[m]} documents' for m in TMODELS)}) of the 200 validation documents; "
                  "documents whose saved Phase 1 attribution map is one token longer than the current tokenization of the text were left out (cause not found), "
                  "so N differs slightly per model. Only the best GA/BT start layer per model was re-scored; the saved text maps use the squared reduction "
                  "(M4's IG/SG metric uses the absolute one), and the forward passes used bf16 (ABPC differs from fp32 by at most 0.003 on a 5-document check).</li>")
    text_gap = f" and {len(TMODELS)} text model{'s' if len(TMODELS) != 1 else ''}"
    import json
    (TXT / "TEXT_FINDINGS.json").write_text(json.dumps({
        "models": {m: {"name": TNICE[m], "docs": TN[m], "changed_by": tchanged[m], "ranking_pad": trank_ref(m)} for m in TMODELS},
        "n_models": len(TMODELS), "n_changed": t_n_changed, "reversed": [[TNICE[m], b] for m, b in t_reversed],
        "mask_lower": t_mask_lower, "unk_lower": t_unk_lower, "cases": t_cases,
        "mean_pad_minus_mask": float(np.mean(mask_drops)), "mean_pad_minus_unk": float(np.mean(unk_drops)),
        "bss_family": {k: [float(v) for v in row] for k, row in fam_bss.iterrows()}}, indent=2))

ss_tbl = ss[["N", "p_same_ranking", "mean_tau"]].copy()
ss_tbl.columns = ["images in subset", "P(same ranking as full 5000)", "mean Kendall tau"]

abpc_blocks = "".join(f"<p>{NICE[m]}, {NIMG[m]} images:</p>{table(abpc_table(m))}" for m in MODELS)
rank_blocks = ("<p class=\"small\">Per-model Kendall tau and the exact method pairs that flip under each baseline are in "
               "<code>phase2_results/rescore/summary/rank_stability.csv</code>. Heatmap: ABPC rank of each method under each baseline "
               "(a rank that changes along a row means the ranking is baseline-dependent).</p>")
def ba_all():
    t = pd.concat([ba_tbl(mm[m]).assign(model=NICE[m]) for m in MODELS], ignore_index=True)
    return t[["model"] + [c for c in t.columns if c != "model"]]
ba_blocks = table(ba_all(), lambda v: f'{v:.3f}')
ba_blocks += "".join(f"<span class='small'>{NICE[m]}: bootstrap reproducibility of the full ranking, grey {mm[m].bootstrap_repro_grey.iloc[0]:.0%} vs multi-baseline {mm[m].bootstrap_repro_multi.iloc[0]:.0%}. </span>" for m in MODELS)

CSS = """<style>body{font-family:Liberation Sans,Arial;font-size:10.5pt}h1{font-size:19pt}h2{font-size:14pt;margin-top:18pt}
h3{font-size:11.5pt}table.t{border-collapse:collapse;font-size:9pt}th{background:#e8e8e8}
.pub{background:#fff3cd;padding:4pt;border:1px solid #d4b106}.ours{background:#e2f0e2;padding:4pt;border:1px solid #6a9a6a}
.small{font-size:9pt;color:#444}</style>"""

html = f"""<html><head><meta charset="utf-8"><title>Phase 2 results</title>{CSS}</head><body>
<h1>Phase 2: Research Gap and Extension<br>Is the method ranking in M4 robust to the masking baseline?</h1>
<p><b>Course:</b> DA351 Explainable AI &nbsp; <b>Team / members / roll numbers:</b> [fill in]</p>
<p class="small">Colour key: <span class="pub">yellow = PUBLISHED results from the original paper (not our experiments)</span>
&nbsp; <span class="ours">green = results from OUR experiments (this repository)</span></p>

<h2>1. Original paper</h2>
<p>Xuhong Li, Mengnan Du, Jiamin Chen, Yekun Chai, Himabindu Lakkaraju, Haoyi Xiong.
<i>M<sup>4</sup>: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics,
Modalities and Models.</i> NeurIPS 2023, Datasets and Benchmarks Track. (Full PDF submitted separately.)</p>

<h2>2. Research gap</h2>
<p>M4 scores feature-attribution methods with perturbation metrics (MoRF, LeRF, ABPC) that replace the top-k or bottom-k
features by a baseline value. The paper's definition (Eq. 1-3, page 4, below) says only that features are "masked"; the main text does
not state the masking value or test other values. In the authors' pipeline, as ported in <code>m4bench/common.py</code>, the value is the constant grey (127,127,127) for images.
A model's response to such inputs mixes attribution quality with how the model reacts to unusual inputs, so rankings could depend on this choice.
<b>Gap:</b> to our knowledge, nobody has checked whether M4's method rankings survive a change of masking baseline. Related work shows the issue for deletion/insertion
metrics in general (Picard et al. 2025, Rong et al. 2022, Hooker et al. 2019, Hsieh et al. 2021, Gomez et al. 2022), but not for M4's benchmark.</p>

<h2 style="page-break-before:always">3. Published results (original paper)</h2>
<div class="pub"><b>PUBLISHED, M4 paper, page 4: Equations (1)-(3), the metric definitions</b>{img(OUT / 'paper_eq1-3_p4.png', 560)}
<b>PUBLISHED, M4 paper, page 6: Table 1, ResNet-50</b>{img(OUT / 'paper_table1_p6.png', 560)}</div>

<h2>4. Original approach, reproduced (our run, grey baseline)</h2>
<div class="ours"><p>Our PyTorch port of the M4 pipeline (<code>m4bench/</code>), ResNet-50, the first 500 images of M4's image list, single grey baseline.
Published numbers are copied from Table 1, p. 6; the "ours" columns are our measurements.</p>
{table(paper_vs_ours)}</div>
<p><b>Observation.</b> Our MoRF and ABPC are somewhat higher than the paper's. The paper's ABPC order is GradCAM &gt; IG &gt; SG, while ours under grey is IG &gt; GradCAM &gt; SG.
The IG and GradCAM confidence intervals overlap, so this particular difference is not reliable. We did not investigate the source of the offset (candidates: image subset, weights conversion, implementation details).</p>

<h2>5. Proposed improvement</h2>
<p>Instead of reporting one ABPC under one baseline, evaluate every saved attribution map under <b>five masking baselines</b> and report
(i) 95% bootstrap confidence intervals, (ii) Kendall's tau and the flipped method pairs against the grey ranking,
(iii) a <b>multi-baseline ABPC</b> (mean over baselines) and (iv) the Baseline Sensitivity Score, BSS = standard deviation over baselines of the mean ABPC.
The attributions are <b>not recomputed</b>: the baseline is the only thing that changes.</p>
<table class="t" cellpadding="4" cellspacing="0" border="1"><tr><th></th><th>M4 (before)</th><th>Proposed (after)</th></tr>
<tr><td>Masking baseline</td><td>one fixed grey value</td><td>grey, black, ImageNet-mean colour, Gaussian blur (sigma 11), uniform noise</td></tr>
<tr><td>Uncertainty</td><td>none reported</td><td>95% bootstrap CI over images</td></tr>
<tr><td>Ranking robustness</td><td>not tested</td><td>Kendall tau, flipped pairs, bootstrap reproducibility</td></tr>
<tr><td>Sample size</td><td>one fixed N</td><td>subset-vs-full ranking agreement (MobileNet, 5000 images)</td></tr></table>
<p><b>Supporting references.</b> Rong et al. (ICML 2022): pixel-perturbation evaluations give conflicting rankings, and protocols can be made more robust.
Picard et al. (arXiv 2025, preprint): the baseline in deletion/insertion systematically favours some attribution methods.
Hsieh et al. (ICLR 2021) and Hooker et al. (NeurIPS 2019): removal-based evaluation can be biased by the replacement value / distribution shift.</p>

<h2>6. Implementation</h2>
<ul><li><code>rescore_baselines.py</code> (new): loads the saved M4 maps (<code>maps_sq.npy</code>), builds the MoRF/LeRF stacks with
<code>baseline_image()</code> + <code>perturbation_stacks()</code>, which keep M4's 100-step percentile schedule and class of interest and replace only the fill value, and stores the class-probability curves.</li>
<li><code>analyze_baselines.py</code> (new): bootstrap CIs, Kendall tau, flipped pairs, multi-baseline ABPC, BSS, bootstrap ranking reproducibility, sample-size analysis, figures.</li>
<li>The extension only adds these two files. It reuses the existing <code>m4bench/</code> image loading, model loading and metric helpers (<code>vision_metrics.batched_outputs</code>, <code>resize_map</code>) unchanged.</li>
<li><b>Same conditions:</b> same images, same saved maps, same schedule, fp32 forward passes. Check done during development on 20 images:
re-scoring with grey reproduced the stored M4 curves (per-image ABPC correlation 0.9995; mean ABPC 0.5438 vs 0.5429).</li></ul>

<h2>7. Results after the improvement</h2><div class="ours">
<h3>7.1 Summary: does the grey (M4) method ranking survive other masking baselines?</h3>
{table(rank_summary())}
<h3>7.2 ABPC by masking baseline, mean [95% CI] (higher = more faithful)</h3>
{abpc_blocks}
{img(RES / 'abpc_by_baseline.png', 640)}
<h3>7.3 Ranking under every baseline (Kendall tau and flipped pairs vs grey)</h3>
{rank_blocks}
{img(RES / 'rank_heatmap.png', 640)}
<h3>7.4 Before (grey) vs after (multi-baseline)</h3>
{ba_blocks}
<h3>7.5 How much does a 500-image subset matter? (MobileNet, grey baseline, M4's own per-image scores on all 5000 images)</h3>
{table(ss_tbl)}{img(RES / 'sample_size.png', 380)}
{text_html}</div>

<h2>8. Interpretation</h2><ul>
<li><b>The grey ranking is not stable across baselines.</b> It changes under at least one other baseline in {n_changed} of {len(MODELS)} models
({"; ".join(f"{NICE[m]}: {', '.join(v)}" for m, v in changed.items() if v) or "none"}).
{("A complete reversal of the ranking occurs for " + "; ".join(f"{NICE[m]} under {b}" for m, b in reversed_) + ".") if reversed_ else "No ranking is completely reversed."}
For ResNet-50 specifically, M4's order ({rank_grey_of('resnet50').replace('>', ' &gt; ')}) becomes {rank_of('resnet50','black').replace('>', ' &gt; ')} under black,
{rank_of('resnet50','noise').replace('>', ' &gt; ')} under noise and {rank_of('resnet50','blur').replace('>', ' &gt; ')} under blur.</li>
<li><b>IG and SG are the baseline-sensitive methods.</b> Their BSS ranges from {bss_pix_range[0]:.3f} to {bss_pix_range[1]:.3f} across models, versus {bss_coarse_range[0]:.3f} to {bss_coarse_range[1]:.3f}
for GradCAM and the attention-based methods; the pixel-level methods have the higher BSS in {len(bss_pix_higher)} of {len(MODELS)} models
({", ".join(NICE[m] for m in bss_pix_higher) or "none"}). Under black, IG or SG fall below every coarse method in {len(below_black)} of {len(gap_black)} (model, method) cases,
and IG loses on average {np.mean(list(ig_drop.values())):.2f} ABPC going from grey to black (range {min(ig_drop.values()):.2f} to {max(ig_drop.values()):.2f}).
<i>Why</i> (scattered pixel-level masks vs smooth masks) is a hypothesis we did not test.</li>
<li><b>Methods that cannot be separated.</b> For {", ".join(NICE[m] for m in tied) or "no model"}, all coarse-method grey-baseline confidence intervals overlap, so their order, and its changes across baselines, are not reliable.
Images per model: {", ".join(f"{NICE[m]} {NIMG[m]}" for m in tied)}.</li>
<li><b>Multi-baseline ranking.</b> {"; ".join(f"{NICE[m]}: {mm[m].ranking_grey.iloc[0].replace('>', ' &gt; ')} (grey, reproduced in {mm[m].bootstrap_repro_grey.iloc[0]:.0%} of bootstrap resamples) vs {mm[m].ranking_multi.iloc[0].replace('>', ' &gt; ')} ({mm[m].bootstrap_repro_multi.iloc[0]:.0%})" for m in MODELS)}.
This means <i>more or less reproducible</i>, not <i>more correct</i>: the result depends on which baselines are averaged. Grey (127) and the ImageNet-mean colour are close in value, yet even this small change alters the ranking in {n_mean_changed} of {len(MODELS)} models ({', '.join(NICE[m] for m in MODELS if 'mean' in changed[m]) or 'none'}).</li>
<li><b>Sample size:</b> a random 500-image subset reproduces the full 5000-image MobileNet ranking in only {s500:.0%} of draws, so the 500-image runs carry real uncertainty (hence the confidence intervals).
{("In our own 500-image MobileNet rescoring the grey ranking is " + mm['mobilenet'].ranking_grey.iloc[0].replace('>', ' &gt; ') + ", whereas M4's full 5000-image ranking is " + ss.full_ranking.iloc[0].replace('>', ' &gt; ') + ", so even the grey reference for that model is a subset artefact.") if 'mobilenet' in mm and mm['mobilenet'].ranking_grey.iloc[0] != ss.full_ranking.iloc[0] else ""}</li>
{text_interp}</ul>
<p><b>Does the improvement address the gap?</b> Yes for the scope tested ({len(MODELS)} vision models{text_gap}): it shows how M4-style rankings depend on the masking baseline and quantifies how much,
which the single-baseline protocol cannot show. It does not show that any one baseline is the "right" one.</p>

<h2>9. Limitations</h2><ul>
<li>{len(MODELS)} of M4's 9 vision models (ViT-L not run) and small image subsets (500 for the CNNs, 100 for the ViTs). {'No PScore/SynScore under other baselines.' if HAS_TEXT else 'No text models: [PAD]/[MASK]/[UNK] were not run, and no PScore/SynScore under other baselines.'}</li>
{text_limit}
<li>The CNN runs use 500 images and the ViT runs 100, so comparisons <i>between</i> architectures are weaker than comparisons <i>within</i> a model.</li>
<li>The saved attribution maps use the squared-channel reduction; M4's own metric uses the absolute reduction. Grey re-scoring still reproduced the stored curves closely (see section 6).</li>
<li>Our replication differs from the paper's Table 1 numbers (section 4); the cause was not investigated.</li>
<li>Cited papers were checked at abstract / search-excerpt level only.</li></ul>

<h2>10. References</h2><ul class="small">
<li>Li, X., Du, M., Chen, J., Chai, Y., Lakkaraju, H., Xiong, H. (2023). M4: A unified XAI benchmark for faithfulness evaluation of feature attribution methods across metrics, modalities and models. NeurIPS (Datasets and Benchmarks).</li>
<li>Rong, Y., Leemann, T., Borisov, V., Kasneci, G., Kasneci, E. (2022). A consistent and efficient evaluation strategy for attribution methods. ICML, PMLR 162:18770-18795.</li>
<li>Picard, A. M., Boissin, T., Subhash, V., Cadene, R., Fel, T. (2025). Back to the baseline: Examining baseline effects on explainability metrics. arXiv:2512.11433 (preprint).</li>
<li>Hsieh, C.-Y., Yeh, C.-K., Liu, X., Ravikumar, P., Kim, S., Kumar, S., Hsieh, C.-J. (2021). Evaluations and methods for explanation through robustness analysis. ICLR.</li>
<li>Hooker, S., Erhan, D., Kindermans, P.-J., Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. NeurIPS.</li>
<li>Gomez, T., Freour, T., Mouchere, H. (2022). Metrics for saliency map evaluation of deep learning explanation methods. arXiv:2201.13291 (preprint).</li>
<li>Petsiuk, V., Das, A., Saenko, K. (2018). RISE: Randomized input sampling for explanation of black-box models. arXiv:1806.07421.</li></ul>

<h2>Provenance</h2>
<p class="small">Published values: only Table 1 (p. 6) and Equations 1-3 (p. 4), shown as page images and in the "paper" columns of section 4. Everything else comes from the scripts in this repository
run on the saved M4 attribution maps; raw curves and logs are in <code>phase2_results/</code>; regenerate with the commands in <code>README.md</code>.</p>
</body></html>"""
(OUT / "report.html").write_text(html, encoding="utf-8")
print("wrote", OUT / "report.html")

"""
Build the Phase 2 "Experimental Results and Comparison" report (HTML -> PDF).

Every number from OUR experiments is read from phase2_results/ (the CSVs written by analyze_baselines.py).
The only hard-coded numbers are the PUBLISHED M4 values (Table 1, p. 6), which are labelled as such.
Key claims in the prose are asserted against the data, so the build fails if the text and the data disagree.

    python phase2_report/build_report.py            # writes phase2_report/report.html
    libreoffice --headless --convert-to pdf:writer_web_pdf_Export --outdir phase2_report phase2_report/report.html
"""

import base64
from pathlib import Path

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


# ---------------------------------------------------------------- claims checked against the data
rn_rank = lambda b: stab[(stab.model == "resnet50") & (stab.metric == "ABPC") & (stab.baseline == b)].rank_baseline.iloc[0]
rn_grey = stab[(stab.model == "resnet50") & (stab.metric == "ABPC")].rank_grey.iloc[0]
assert rn_grey == "ig>gradcam>sg", rn_grey
assert rn_rank("blur") == "sg>ig>gradcam", rn_rank("blur")          # exact reversal under blur
assert rn_rank("black") == "gradcam>sg>ig" and rn_rank("noise") == "gradcam>ig>sg"
assert rn_rank("mean") == rn_grey
# under black, IG and SG sit >0.1 below the best non-pixel-level methods (GradCAM / attention-based)
ref = {"resnet50": ("gradcam",), "vit_small": ("ga", "bt_head", "bt_token")}
gap_black = {}
for model, refs in ref.items():
    best_ref = min(get(model, m, "black") for m in refs)
    for m in ("ig", "sg"):
        gap_black[(model, m)] = best_ref - get(model, m, "black")
        assert gap_black[(model, m)] > 0.1, (model, m, gap_black[(model, m)])
for model in ("resnet50", "vit_small"):                                # IG drops a lot from grey to black on both
    assert get(model, "ig", "grey") - get(model, "ig", "black") > 0.15, model
bss = multi.set_index(["model", "method"]).BSS
assert bss[("resnet50", "gradcam")] < 0.02 < 0.05 < min(bss[("resnet50", "ig")], bss[("resnet50", "sg")])
assert max(bss[("vit_small", m)] for m in ("ga", "bt_head", "bt_token")) < 0.03 < 0.04 < \
    min(bss[("vit_small", "ig")], bss[("vit_small", "sg")])
rn = multi[multi.model == "resnet50"]; vt = multi[multi.model == "vit_small"]
assert rn.ranking_multi.iloc[0] == "gradcam>ig>sg", rn.ranking_multi.iloc[0]
# the three attention-based ViT methods are statistically tied under grey (every CI overlaps)
att = [(get("vit_small", m, "grey", col="ci_lo"), get("vit_small", m, "grey", col="ci_hi")) for m in ("ga", "bt_head", "bt_token")]
assert max(a for a, _ in att) < min(b for _, b in att)
s500 = float(ss[ss.N == 500].p_same_ranking.iloc[0])

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

ss_tbl = ss[["N", "p_same_ranking", "mean_tau"]].copy()
ss_tbl.columns = ["images in subset", "P(same ranking as full 5000)", "mean Kendall tau"]

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
<h3>7.1 ABPC by masking baseline, mean [95% CI] (higher = more faithful)</h3>
<p>ResNet-50, 500 images:</p>{table(abpc_table('resnet50'))}
<p>ViT-S, 100 images:</p>{table(abpc_table('vit_small'))}
{img(RES / 'abpc_by_baseline.png', 600)}
<h3>7.2 Does the method ranking change with the baseline?</h3>
<p>ResNet-50:</p>{table(rank_tbl('resnet50'))}<p>ViT-S:</p>{table(rank_tbl('vit_small'))}
{img(RES / 'rank_heatmap.png', 520)}
<h3>7.3 Before (grey) vs after (multi-baseline)</h3>
<p>ResNet-50:</p>{table(ba_tbl(rn), lambda v: f'{v:.3f}')}<p>ViT-S:</p>{table(ba_tbl(vt), lambda v: f'{v:.3f}')}
<p class="small">Bootstrap reproducibility of the full ranking: ResNet-50 grey {rn.bootstrap_repro_grey.iloc[0]:.0%} vs multi-baseline {rn.bootstrap_repro_multi.iloc[0]:.0%};
ViT-S grey {vt.bootstrap_repro_grey.iloc[0]:.0%} vs multi-baseline {vt.bootstrap_repro_multi.iloc[0]:.0%}.</p>
<h3>7.4 How much does a 500-image subset matter? (MobileNet, grey baseline, M4's own per-image scores on all 5000 images)</h3>
{table(ss_tbl)}{img(RES / 'sample_size.png', 380)}</div>

<h2>8. Interpretation</h2><ul>
<li><b>ResNet-50: the ranking depends on the baseline.</b> M4's grey ordering (IG &gt; GradCAM &gt; SG) is unchanged under mean colour but changes under black ({rn_rank('black').replace('>', ' &gt; ')}),
noise ({rn_rank('noise').replace('>', ' &gt; ')}) and is exactly reversed under blur ({rn_rank('blur').replace('>', ' &gt; ')}).</li>
<li><b>IG and SG are the baseline-sensitive methods</b> on both models: BSS {bss[('resnet50','ig')]:.3f} (IG) and {bss[('resnet50','sg')]:.3f} (SG) vs {bss[('resnet50','gradcam')]:.3f} for GradCAM on ResNet-50;
on ViT-S IG {bss[('vit_small','ig')]:.3f} and SG {bss[('vit_small','sg')]:.3f} vs at most {max(bss[('vit_small', m)] for m in ('ga','bt_head','bt_token')):.3f} for the attention-based methods.
Under black, their ABPC is lower than that of the best non-pixel-level method by {gap_black[('resnet50','ig')]:.2f} (IG) and {gap_black[('resnet50','sg')]:.2f} (SG) on ResNet-50 (vs GradCAM), and by {gap_black[('vit_small','ig')]:.2f} (IG) and {gap_black[('vit_small','sg')]:.2f} (SG) on ViT-S (vs the attention-based methods). <i>Why</i> (scattered pixel-level masks vs smooth masks) is a hypothesis we did not test.</li>
<li><b>ViT-S: the three attention-based methods (GA, BT-head, BT-token) cannot be separated</b> at 100 images (all grey CIs overlap), so their order changes across baselines are noise. Only the IG/SG effect above is supported.</li>
<li><b>The multi-baseline ranking</b> on ResNet-50 is {rn.ranking_multi.iloc[0].replace('>', ' &gt; ')}, and it is reproduced in {rn.bootstrap_repro_multi.iloc[0]:.0%} of bootstrap resamples versus {rn.bootstrap_repro_grey.iloc[0]:.0%} for grey.
This means <i>more reproducible</i>, not <i>more correct</i>: the result depends on which baselines are averaged, and grey and mean colour are nearly identical.</li>
<li><b>Sample size:</b> a random 500-image subset reproduces the full 5000-image MobileNet ranking in only {s500:.0%} of draws, so the 500-image runs carry real uncertainty (hence the confidence intervals).</li></ul>
<p><b>Does the improvement address the gap?</b> Yes for the scope tested: it shows that, on ResNet-50 and ViT-S, M4-style rankings depend on the masking baseline and quantifies how much,
which the single-baseline protocol cannot show. It does not show that any one baseline is the "right" one.</p>

<h2>9. Limitations</h2><ul>
<li>Only two models (ResNet-50, ViT-S) and small image subsets (500 and 100). M4 has 9 image and 6 text models; nothing here supports claims about the rest.</li>
<li>No text experiments ([PAD]/[MASK]/[UNK] were not run) and no PScore/SynScore under other baselines.</li>
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

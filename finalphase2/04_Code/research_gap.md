# Research Gap Identification — Phase 2
## M4 XAI Benchmark Replication (NeurIPS 2023)
**Course: DA351 – Explainable AI (XAI)**

---

## 1. Overview of the Original Paper

**M⁴: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods Across Metrics, Modalities and Models** (NeurIPS 2023, Datasets and Benchmarks Track) introduces a comprehensive benchmark that evaluates the *faithfulness* of feature attribution methods using three families of metrics:

- **MoRF / ABPC** — perturbation-based: progressively mask features and measure prediction drop
- **PScore** — model-based: requires a ground-truth attribution-aware model
- **INFD** — distribution-based: information-theoretic measure of feature independence
- **SynScore** — auxiliary model trained on attributions

The benchmark covers 15 models across image and NLP modalities (MobileNet, VGG16, ResNet50/101/152, ViT variants, BERT variants) and 6 XAI methods (BT/H, BT/T, GA, IG, SG, LIME, GradCAM).

---

## 2. Identified Research Gap

> **The M4 paper employs a *fixed, arbitrary masking baseline* for perturbation-based evaluation (grey pixel `(127,127,127)` for vision, `[PAD]` token for NLP), yet never evaluates how sensitive the reported faithfulness metrics are to this design choice. Importantly, M4 never investigates whether its conclusions survive a change of masking baseline.**

### Why This Matters

The perturbation-based faithfulness metrics (MoRF, ABPC) work by progressively replacing features (pixels or tokens) with a *baseline value* and measuring the resulting change in model prediction. The key problem is:

1. **Out-of-Distribution (OOD) artifact**: Replacing pixels with a grey constant or tokens with `[PAD]` creates inputs that never appear during model training. The model's response to these OOD inputs does not purely reflect attribution quality — it may reflect the model's behaviour under distribution shift.

2. **Fixed choice of baseline**: M4's vision evaluation masks with the constant `(127,127,127)` (the value used by the InterpretDL perturbation code that M4 builds on, and by our port in `m4bench/common.py`). The paper does not justify this value over alternatives (black, mean colour, blurred, noise) or test the alternatives.

3. **Metric inflation/deflation**: Depending on how far the masking baseline is from the training distribution, faithfulness scores (MoRF/ABPC) may be systematically inflated or deflated — making cross-paper comparisons unreliable.

### Narrowing the Novelty Claim

Picard et al. (2025) have already shown that the masking baseline in Deletion/Insertion metrics can systematically favour certain attribution methods and produce contradictory rankings. **The specific gap that remains, to the best of our knowledge (we did not do an exhaustive literature search),** is that no one has tested whether *M4's particular conclusions* — across its specific suite of 15 models, 6 XAI methods, image and text modalities — survive a change of masking baseline. M4 uses the word "baseline" only in the context of random attribution baselines; it never discusses the masking value itself. This paper fills that gap.

---

## 3. Supporting Literature

1. **Sturmfels et al. (2020)** — *"Visualizing the Impact of Feature Attribution Baselines"* (Distill).  
   *Background only*: investigates how the reference-point baseline choice *within Integrated Gradients* affects the computed attributions themselves. This is distinct from the masking baseline used when evaluating faithfulness via pixel perturbation.

2. **Hooker et al. (2019)** — *"A Benchmark for Interpretability Methods in Deep Neural Networks (ROAR)"* (NeurIPS).  
   Shows that pixel-replacement evaluation measures a combination of attribution quality *and* model OOD robustness; these cannot be disentangled without careful baseline selection. ROAR addresses the retraining confound but does not study masking-baseline sensitivity per se.

3. **Hsieh et al. (2021)** — *"Evaluations and Methods for Explanation through Robustness Analysis"* (ICLR).  
   Proposes evaluating attributions via adversarial perturbations, motivated by the observation that removal-based methods can be biased by the choice of replacement value. Supports the gap: the masking value is a source of systematic bias in standard perturbation-based evaluation.  
   *(Note: the paper's focus is adversarial robustness, not in-distribution vs. out-of-distribution masking; cite accordingly.)*

4. **Rong et al. (2022)** — *"A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD)"* (ICML).  
   Shows that pixel-perturbation evaluations produce conflicting rankings of attribution methods across protocols, naming information leaking through the shape of removed regions as a confound. Supports both the gap and the proposed method: demonstrates that evaluation protocols can be made more robust, which is what this work proposes for the M4 setting.

5. **Picard et al. (2025)** — *"Back to the Baseline: Examining Baseline Effects on Explainability Metrics"* (arXiv, preprint — not peer-reviewed).  
   Closest prior work to the proposed contribution. Shows that the baseline choice in Deletion/Insertion metrics systematically favours certain attribution methods and that even a linear model can receive contradictory rankings under different baselines; proposes a model-dependent baseline. Directly supports the research gap. **Novelty statement must acknowledge this work**; the remaining contribution is testing M4's specific conclusions across its full model and modality scope.

6. **Gomez et al. (2022)** — *"Metrics for Saliency Map Evaluation of Deep Learning Explanation Methods"* (arXiv, preprint — not peer-reviewed).  
   Argues that computing deletion/insertion AUC (DAUC/IAUC) shows the model images that are out of its training distribution, which may make the explained model behave unreliably. Provides additional context for the gap (OOD masking as a confound).

7. **Petsiuk et al. (2018)** — *"RISE: Randomized Input Sampling for Explanation"* (BMVC).  
   Origin of the deletion/insertion metrics that MoRF/ABPC resemble. Cite as background when introducing the evaluation protocol; verify the exact masking value used in RISE before citing specific details.

---

## 4. Proposed Extension: Baseline Sensitivity Analysis

We propose quantifying the *Baseline Sensitivity Score (BSS)* — the standard deviation of faithfulness metrics (MoRF, ABPC) across different masking baselines for the same attribution method.

### Vision Modality Baselines (`rescore_baselines.py`)
| Baseline | Description | Expected relation to data |
|---|---|---|
| **Grey** `(127,127,127)` | M4 paper default ("before" reference) | Off-distribution constant |
| **Black** `(0,0,0)` | Zero-pixel | Off-distribution constant |
| **Mean** | ImageNet mean colour | Off-distribution constant (close to grey) |
| **Gaussian Blur** (σ = 11) | Blurred copy of the same image | Closer to the image's own statistics |
| **Noise** | Uniform per-pixel noise, seeded per image | Off-distribution |

The attribution maps are **not recomputed**: the saved M4 maps (`results/m4/vision/<model>/<method>/maps_sq.npy`) are re-masked under each baseline, so the baseline is the only thing that changes. Re-scoring with grey reproduces M4's stored curves (per-image ABPC correlation 0.9995 on a 20-image check).

### NLP Modality Baselines (planned, not yet implemented)
`[PAD]` (M4 default), `[MASK]`, `[UNK]`. Not part of the current implementation; see Section 6.

### Proposed protocol (the "improvement") vs. M4
| | M4 (before) | Proposed (after) |
|---|---|---|
| Masking baseline | one fixed grey value | five baselines, with multi-baseline ABPC = mean over baselines |
| Uncertainty | none reported | 95% bootstrap CI over images |
| Ranking robustness | not tested | Kendall τ vs. the grey ranking; list of flipped method pairs; bootstrap reproducibility of the ranking |
| Sample size | single fixed N | subset-vs-full ranking agreement (MobileNet, 5000 images) |

### Baseline Sensitivity Score (BSS)
```
BSS(method) = std over baselines of [ mean over images of ABPC(method, baseline) ]
BSS_per_image(method) = mean over images of [ std over baselines of ABPC ]
```
A high BSS indicates the faithfulness score depends strongly on the choice of masking baseline. A high BSS alone does not show that the *ranking* of methods changes; that is tested separately with Kendall τ and flipped pairs.

---

## 5. Expected Findings

Based on prior literature, we hypothesize:

1. **Grey masking will produce systematically different MoRF/ABPC scores** compared to in-distribution baselines (blur for vision, [MASK] for NLP).
2. **Methods differ in how baseline-sensitive they are.** BSS is a property of a method (spread across baselines), not of a single baseline, so the hypothesis is stated per method: attributions whose masks are scattered high-frequency pixel patterns (IG, SG) are expected to be more sensitive than smooth or coarse ones (GradCAM, attention-based). The mechanism is not tested here.
3. **ABPC will be more sensitive to baseline choice than MoRF** because ABPC depends on both the MoRF *and* LeRF orderings, compounding any OOD effect.
4. **Rankings of XAI methods will shift** across baselines, echoing the conflicting-rankings finding of Rong et al. (2022) and Picard et al. (2025) in the specific M4 setting.

---

## 6. Scope

**Implemented:**
- **Image models**: ResNet-50, ResNet-101, ResNet-152, VGG16 and MobileNet (GradCAM, IG, SG; first 500 ImageNet-val images of M4's list) and ViT-S, ViT-B, MAE-ViT-B (IG, SG, GA, BT-head, BT-token; first 100 images).
- **Text models**: DistilBERT, BERT-base, BERT-large, ERNIE-2.0-base, RoBERTa-base (IG, SG, LIME and the best GA / BT-head / BT-token start layer; the 200 MovieReview validation documents whose saved map matches the tokenization: DistilBERT 188, BERT-base 188, BERT-large 197, ERNIE-2.0-base 188, RoBERTa-base 189) re-scored with [PAD] (M4), [MASK] and [UNK] by `rescore_text_baselines.py`.
- **Sample-size analysis**: MobileNet, using M4's own grey-baseline per-image scores on all 5000 images (no re-scoring).
- **Check**: re-scoring with grey reproduces M4's stored per-image ABPC for every model and method (correlation 0.992 to 1.000).
- **Baselines**: the five vision baselines above.

**Not done (stated limitations):** ViT-L, PScore/SynScore under other baselines, and re-computing the attributions themselves. Results therefore cover 8 vision models, 5 text models and small subsets, so conclusions about ViT-L cannot be drawn from this work.

We cannot claim broader coverage than Picard et al. (2025): we have read only its abstract and do not know its full scope.

---

## 7. Significance

This extension is directly actionable for the community:
- **Future XAI benchmarks** should standardize or explicitly report the masking baseline used.
- **Practitioners** comparing faithfulness scores across papers must account for baseline differences.
- **A low-cost protocol**: averaging over several masking baselines and reporting confidence intervals needs only extra forward passes, no retraining. Whether it actually makes rankings more stable is an empirical question answered in the results, not assumed here.

---

## References

> [!WARNING]
> All references below were verified from **abstracts and search excerpts only** — full PDFs have not been read. Read each paper before citing specific claims or numbers in the final report.

- Li, X., Du, M., Chen, J., Chai, Y., Lakkaraju, H., & Xiong, H. (2023). M⁴: A unified XAI benchmark for faithfulness evaluation of feature attribution methods across metrics, modalities and models. *NeurIPS 2023, Datasets and Benchmarks Track*.
- Hooker, S., Erhan, D., Kindermans, P. J., & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS*.
- Rong, Y., et al. (2022). A consistent and efficient evaluation strategy for attribution methods. *ICML*.
- Picard, et al. (2025). Back to the baseline: Examining baseline effects on explainability metrics. *arXiv* (preprint).
- Hsieh, C. K., et al. (2021). Evaluations and methods for explanation through robustness analysis. *ICLR*.
- Gomez, et al. (2022). Metrics for saliency map evaluation of deep learning explanation methods. *arXiv* (preprint).
- Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized input sampling for explanation of black-box models. *BMVC*.
- Sturmfels, P., Lundberg, S., & Lee, S. I. (2020). Visualizing the impact of feature attribution baselines. *Distill* (background only — concerns IG's internal baseline, not masking baselines for evaluation).
- Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K. R. (2017). Evaluating the visualization of what a deep neural network has learned. *IEEE TNNLS*.

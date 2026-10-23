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

> **The M4 paper employs a *fixed, arbitrary masking baseline* for perturbation-based evaluation (grey pixel `(127,127,127)` for vision, `[PAD]` token for NLP), yet never evaluates how sensitive the reported faithfulness metrics are to this design choice.**

### Why This Matters

The perturbation-based faithfulness metrics (MoRF, ABPC) work by progressively replacing features (pixels or tokens) with a *baseline value* and measuring the resulting change in model prediction. The key problem is:

1. **Out-of-Distribution (OOD) artifact**: Replacing pixels with a grey constant or tokens with `[PAD]` creates inputs that never appear during model training. The model's response to these OOD inputs does not purely reflect attribution quality — it may reflect the model's behaviour under distribution shift.

2. **Arbitrary choice of baseline**: The grey `(127,127,127)` pixel was chosen by the M4 paper because it is the InterpretDL library default. There is no theoretical justification for this choice over alternatives (black, blurred, noise).

3. **Metric inflation/deflation**: Depending on how far the masking baseline is from the training distribution, faithfulness scores (MoRF/ABPC) may be systematically inflated or deflated — making cross-paper comparisons unreliable.

---

## 3. Supporting Literature

1. **Sturmfels et al. (2020)** — *"Visualizing the Impact of Feature Attribution Baselines"* (Distill).  
   Demonstrates that the choice of baseline in Integrated Gradients fundamentally changes which features are considered important. A poorly chosen baseline can make arbitrary features appear highly attributed.

2. **Hooker et al. (2019)** — *"A Benchmark for Interpretability Methods in Deep Neural Networks (ROAR)"* (NeurIPS).  
   Shows that pixel-replacement with a constant value (a common evaluation strategy) measures a combination of attribution quality *and* model OOD robustness — these cannot be disentangled without careful baseline selection.

3. **Hsieh et al. (2021)** — *"Evaluations and Methods for Explanation through Robustness Analysis"* (ICLR).  
   Proposes that in-distribution perturbations (e.g., blurred regions) lead to more reliable faithfulness estimates compared to constant or random replacement.

---

## 4. Proposed Extension: Baseline Sensitivity Analysis

We propose quantifying the *Baseline Sensitivity Score (BSS)* — the standard deviation of faithfulness metrics (MoRF, ABPC) across different masking baselines for the same attribution method.

### Vision Modality Baselines (ResNet-50 + IG)
| Baseline | Description | In/Out of Dist. |
|---|---|---|
| **Grey** `(127,127,127)` | M4 paper default | Out-of-distribution |
| **Black** `(0,0,0)` | Zero-pixel | Out-of-distribution |
| **Gaussian Blur** | Blurred version of image | In-distribution |
| **Random Noise** | Gaussian noise | Out-of-distribution |

### NLP Modality Baselines (BERT-base + IG)
| Baseline | Description | In/Out of Dist. |
|---|---|---|
| **[PAD]** | M4 paper default | Out-of-distribution |
| **[MASK]** | BERT's native masked token | In-distribution (pre-training) |
| **[UNK]** | Unknown token | Out-of-distribution |

### New Metric: Baseline Sensitivity Score (BSS)
```
BSS(method, metric) = std(metric_score across baselines) per image
Mean BSS = mean over all images
```
A high BSS indicates the faithfulness metric is not robust to the choice of masking baseline, calling into question whether the reported scores in the M4 paper accurately reflect attribution method quality.

---

## 5. Expected Findings

Based on prior literature, we hypothesize:

1. **Grey masking will produce systematically different MoRF/ABPC scores** compared to in-distribution baselines (blur for vision, [MASK] for NLP).
2. **Gaussian blur and [MASK] token** will give more consistent (lower BSS) scores because they keep the masked regions closer to the training distribution.
3. **ABPC will be more sensitive to baseline choice than MoRF** because ABPC depends on both the MoRF *and* LeRF orderings, compounding any OOD effect.

---

## 6. Significance

This extension is directly actionable for the community:
- **Future XAI benchmarks** should standardize or explicitly report the masking baseline used.
- **Practitioners** comparing faithfulness scores across papers must account for baseline differences.
- **A simple fix**: using a Gaussian blur or learned inpainting baseline can improve the reliability of perturbation-based faithfulness evaluation at no additional model training cost.

---

## References

- Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K. R. (2017). Evaluating the visualization of what a deep neural network has learned. *IEEE TNNLS*.
- Sturmfels, P., Lundberg, S., & Lee, S. I. (2020). Visualizing the impact of feature attribution baselines. *Distill*.
- Hooker, S., Erhan, D., Kindermans, P. J., & Kim, B. (2019). A benchmark for interpretability methods in deep neural networks. *NeurIPS*.
- Hsieh, C. K., et al. (2021). Evaluations and methods for explanation through robustness analysis. *ICLR*.
- Zhou, X., et al. (2023). M⁴: A unified XAI benchmark for faithfulness evaluation. *NeurIPS*.

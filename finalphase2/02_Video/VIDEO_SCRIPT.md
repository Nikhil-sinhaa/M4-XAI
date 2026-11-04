# Phase 2 video: FINAL spoken script with display sequence (10 minutes max)

Two PDFs are used: **[PAPER]** = `01_Original_Paper/M4_NeurIPS2023_paper.pdf`, **[REPORT]** = `03_Results_and_Comparison_PDF/Phase2_Results_and_Comparison.pdf` (15 pages).
If you edit the report (for example to add names), re-check the page numbers below.
Text after SAY is what you speak; adapt the wording so it sounds like you. About 1,360 words, about 9 minutes at 150 words per minute.

## Display sequence at a glance
| # | Time | Show |
|---|---|---|
| 1 | 0:00 | [PAPER] page 1 |
| 2 | 1:00 | [PAPER] page 4 (Eq. 1-3), then page 6 (Table 1) |
| 3 | 3:00 | [REPORT] page 3 (section 5 Proposed improvement, section 6 Implementation) |
| 4 | 5:00 | Editor: `rescore_baselines.py`, then `analyze_baselines.py`, then terminal |
| 5 | 7:00 | [REPORT] pages 2-3 (original reproduced + 7.1 summary), page 7 (figures), page 8 (7.4 before/after), page 12 (text figures), page 9 (sample size) |
| 6 | 8:30 | [REPORT] page 14 (8 Interpretation) and page 15 (9 Limitations) |

---

## 1. Introduction (about 1:00)  |  SHOW: [PAPER] page 1
SAY:
"Hello, this is our Phase 2 presentation for DA351, Explainable AI. Our paper is *M4: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods across Metrics, Modalities and Models*, by Xuhong Li, Mengnan Du, Jiamin Chen, Yekun Chai, Himabindu Lakkaraju and Haoyi Xiong, from NeurIPS 2023, Datasets and Benchmarks track.

The problem it tackles is this: methods like Integrated Gradients, SmoothGrad and GradCAM all produce different heatmaps for the same prediction, and there is no ground truth to say which one is right. M4 compares them with several faithfulness metrics, on image and text models.

Our objective in Phase 2 is to test whether M4's conclusions still hold when we change one design choice that the paper fixes: the masking baseline."

## 2. Research gap (about 2:00)  |  SHOW: [PAPER] page 4, point at Eq. 1-3 and the words "features are masked"; then [PAPER] page 6, Table 1
SAY:
"First, the metrics. MoRF, most relevant first, deletes the pixels the explanation calls most important, step by step, and measures how fast the model's probability drops. LeRF deletes the least important pixels first. ABPC is the area between the two curves, so a higher ABPC means a more faithful explanation. These are equations one to three here on page 4.

Notice the words 'features are masked'. To delete a pixel you have to replace it with some value, but the paper's main text never says which value, and never tests another. In the authors' pipeline, which we ported, it is a constant grey, 127, 127, 127. [switch to page 6] This is Table 1, ResNet-50. Everything the benchmark says about which method is best rests on numbers like these.

Why does this matter? A grey patch is not something the model saw in training. So when the probability drops, we can't tell whether it dropped because important information was removed, or because the model was confused by the grey patch. If different masking values give different rankings, then the ranking partly reflects the baseline, not the methods.

Related work points the same way. Hooker and colleagues, in 2019, discuss distribution shift when removing features. Hsieh and colleagues, ICLR 2021, argue removal-based evaluation can be biased by the replacement value. Rong and colleagues, ICML 2022, show pixel-perturbation evaluations can give conflicting rankings. And Picard and colleagues, in a 2025 preprint, show the baseline in deletion and insertion metrics can favour some methods.

To be honest about novelty: the general problem is known. What nobody has tested, to our knowledge, is whether M4's own conclusions survive a change of baseline. That is our gap."

## 3. Proposed improvement (about 2:00)  |  SHOW: [REPORT] page 3, section 5 (Proposed improvement)
SAY:
"Our improvement is a more robust evaluation protocol. In the original approach, each method gets one ABPC number, under one grey baseline, with no uncertainty. In ours, we evaluate every explanation under five masking baselines: grey, which is M4's default; black; the ImageNet mean colour; a blurred copy of the same image, which stays closer to the image's own statistics; and random noise.

On top of that we report four things. One: 95 percent bootstrap confidence intervals, so we know which differences are real. Two: Kendall's tau, and which pairs of methods swap places compared with the grey ranking. Three: a multi-baseline ABPC, the average over the five baselines. Four: the Baseline Sensitivity Score, BSS, which is the standard deviation across baselines of a method's mean ABPC. A high BSS means the method's score depends heavily on the baseline.

One important design decision: we do not recompute the explanations. Integrated Gradients costs hundreds of forward and backward passes per image. M4's run already saved the attribution maps, so we reuse them and only change how pixels are masked. That keeps the study affordable, and it makes the comparison fair, because the baseline is the only thing that changes.

The idea of making evaluation more robust follows Rong and Picard. What differs from the original paper is that we measure how stable the ranking is, instead of reporting one score."

## 4. Implementation (about 2:00)  |  SHOW: editor `rescore_baselines.py`, then `analyze_baselines.py`, then the terminal
SAY:
"We added two new files and left M4's pipeline as it is. In rescore_baselines.py, the function baseline_image builds the fill image for each baseline: constant grey, black, the mean colour, a Gaussian blur with sigma eleven, or uniform noise seeded per image.

The function perturbation_stacks follows M4's schedule exactly: a hundred steps, with thresholds taken from percentiles of the attribution map. The masks are nested, so each step is one numpy where call: where the pixel is among the most important, use the baseline, otherwise keep the original. LeRF does the same from the least important side.

Then rescore_method sends these images through the model in batches, takes the probability of the class the model predicted on the clean image, as M4 does, and saves the MoRF and LeRF curves.

[switch to analyze_baselines.py] Here we compute MoRF and ABPC from the curves using the paper's definitions. Then bootstrap confidence intervals, Kendall's tau and flipped pairs, the multi-baseline score, BSS, and how often a bootstrap resample reproduces the same ranking.

To check that before and after are comparable, we re-scored with grey and compared against M4's stored curves on twenty images. The per-image ABPC correlation was 0.9995. One caveat: the saved maps use a squared-channel reduction while M4's metric uses the absolute one, and the check shows the difference is small.

[switch to terminal, run the analysis command, show the output] Everything regenerates from the repository without a GPU, from the committed curves."

Command to run on screen:
`python analyze_baselines.py --rescore_root ../05_Results_Data/rescore --vision_root "" --out /tmp/demo` (see 02_Video/terminal_demo_command.txt)

## 5. Experimental results (about 2:00)  |  SHOW: [REPORT] pages 2-3, then 7, then 8, then 12 (text), then 9
SAY:
"We ran five CNNs on 500 images with GradCAM, IG and SmoothGrad, and three vision transformers on 100 images with five methods. As a check, re-scoring under grey reproduces M4's own stored scores with correlation above 0.99, so the baseline is the only thing that changes.

[pages 2 and 3] First, the original approach. Under grey on ResNet-50, our numbers are a bit higher than the paper's Table 1, and our ABPC order is IG, GradCAM, SG, where the paper has GradCAM first. But the confidence intervals of IG and GradCAM overlap, so that difference is not reliable, and we did not investigate the offset. The summary table here is the main result: in all eight models, the grey ranking changes under at least one other baseline.

[page 7, bar chart] Each colour is a baseline, and the outlined bar is grey, the original. In every model, GradCAM and the attention-based methods barely move, but IG and SG drop sharply under black. On ResNet-50, IG falls from 0.466 under grey to 0.287 under black.

[page 7, heatmap] This shows each method's rank. On ResNet-50: under grey, IG, GradCAM, SG. Under black, GradCAM, SG, IG. Under blur it is exactly reversed. ResNet-152 under blur and MobileNet under black and noise reverse too. VGG16 is the most stable.

[page 8] The BSS agrees. IG and SG score between 0.03 and 0.09, against at most 0.03 for GradCAM and the attention-based methods, and they are higher in every one of the eight models. But on the three transformers, at only 100 images, the attention-based methods cannot be told apart, so their order changes there are noise.

The multi-baseline ranking is reproducible for the ResNets, but only 18 percent for ViT-B. More reproducible is not the same as more correct.

[page 12] We did the same for text, on five models: DistilBERT, BERT-base, BERT-large, ERNIE and RoBERTa, on about 190 movie-review documents each. M4 masks with the [PAD] token. We re-scored the same saved token attributions with [MASK] and [UNK] as well. The ranking changed in all five models, and ABPC was lower under [MASK] than under [PAD] in 30 of 30 method and model cases, by 0.13 on average. One plausible reason is that [PAD] tokens are removed from attention, while [MASK] and [UNK] stay visible, but we did not test that. One caveat: for text, IG and SG use the squared reduction of the saved maps, so their absolute numbers differ from M4's. The comparison across tokens is still fair, because the maps are the same.

[page 9] Finally, sample size. With MobileNet on all 5000 images, a random 500-image subset reproduces the full ranking only 85 percent of the time. Small subsets carry real uncertainty, which is why we report intervals."

## 6. Conclusion (about 1:00)  |  SHOW: [REPORT] page 14 (Interpretation), then page 15 (Limitations)
SAY:
"To conclude. Across eight vision models, the ranking of attribution methods under M4's metrics depends on the masking baseline, mainly because Integrated Gradients and SmoothGrad are baseline-sensitive. Reporting one grey number hides this.

Our contribution is a cheap protocol, using only forward passes, with confidence intervals, ranking-stability measures and a multi-baseline score, plus this finding.

The limitations: vision and text models only, on small subsets. We did not run ViT-Large. We do not know why IG and SG are sensitive; our guess is that their masks are scattered pixels, but we did not test it. And we cannot say which baseline is the right one.

Future work is ViT-Large, and comparing against retraining-based evaluation. Thank you."

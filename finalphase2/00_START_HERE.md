# finalphase2: everything for the Phase 2 submission

Course DA351 Explainable AI. Paper: M4 (NeurIPS 2023). Research gap: **is M4's method ranking robust to the masking baseline?**

## What to submit (Google Form) and where it is
| Form item | File in this folder |
|---|---|
| A. Video (max 10 min, Drive link) | record it yourself using `02_Video/VIDEO_SCRIPT.md` |
| B. Original paper PDF | `01_Original_Paper/M4_NeurIPS2023_paper.pdf` |
| C. Results and comparison PDF | `03_Results_and_Comparison_PDF/Phase2_Results_and_Comparison.pdf` (15 pages) |
| D. Source code + README + GitHub link | `04_Code/` (new/changed files below); GitHub repo `https://github.com/shhadowpdf/m4_xai` |

## Before you submit: things only you can do
1. Fill in team names and roll numbers on page 1 of the report PDF ("[fill in]"). Edit `phase2_report/build_report.py`, rebuild, and recopy the PDF.
2. Record the video (voice on, under 10:00), upload to Drive, set link access to "anyone with the link".
3. Push to GitHub (I have not committed or pushed anything).
4. Read the abstracts of the cited papers once before presenting them as support.

## Rubric to evidence
| Evaluation point | Where it is |
|---|---|
| Research gap, with literature | `04_Code/research_gap.md`, report sections 1-3, video section 2 |
| Proposed improvement + supporting reference | report section 5 (Rong 2022, Picard 2025, Hsieh 2021), video section 3 |
| Implementation | `04_Code/rescore_baselines.py` (vision), `rescore_text_baselines.py` (text), `analyze_baselines.py` (statistics); video section 4 |
| Original vs improved, comparable conditions | report sections 4 (original, grey) and 7 (after); grey re-scoring reproduces M4's stored scores (corr. 0.99+) |
| Published vs own results kept apart | report section 3 is labelled PUBLISHED with paper page numbers (p. 4 Eq. 1-3, p. 6 Table 1); everything else is our run |
| Interpretation and limitations | report sections 8-9, video section 6 |

## New or changed code (identify these in the video and README)
- NEW `rescore_baselines.py`: vision re-scoring under 5 masking baselines (grey, black, mean, blur, noise)
- NEW `rescore_text_baselines.py`: text re-scoring under [PAD], [MASK], [UNK]
- NEW `analyze_baselines.py`: bootstrap CIs, Kendall tau and flipped pairs, multi-baseline ABPC, BSS, ranking reproducibility, sample-size analysis
- NEW `build_report.py`, `finalize_text_docs.py`: report and docs generation
- Unchanged M4 port: `m4bench/`

## Headline numbers to quote (from `05_Results_Data/*/summary/*.csv`)
- Vision, 8 models: the grey ranking changes under at least one other baseline in 8 of 8 models. IG and SG are the sensitive methods (BSS 0.03-0.09 vs at most 0.03 for GradCAM and attention-based). ResNet-50 IG ABPC: 0.466 (grey) to 0.287 (black).
- Text, 5 models: the [PAD] ranking changes under [MASK] or [UNK] in 5 of 5 models. ABPC is lower under [MASK] in 30 of 30 cases (mean 0.130) and under [UNK] in 28 of 30 (mean 0.065).

## Stated limitations (say these in the video)
- CNNs 500 images, ViTs 100 images, text 188-197 documents (documents whose saved map no longer matches the tokenization were left out).
- Saved maps use the squared reduction; M4's own IG/SG metric used the absolute one.
- ViT attention-based methods are statistically tied at 100 images; their order changes are noise.
- ResNet-50 ABPC order differs from the paper's Table 1 (IG/GradCAM intervals overlap); source of the offset not investigated.
- SynScore ran at reduced settings; 3 of 8 models did not learn the task, so those SynScores are not meaningful. SynScore is not part of the research gap.
- No ViT-L. We do not know why IG/SG are baseline-sensitive.

## Still running (not in this folder)
A 5,000-image rerun for four CNNs (MobileNet, VGG16, ResNet-101 done; ResNet-50 IG/SG failed on a laptop GPU fault). It would only tighten confidence intervals. The 500/100-image results in this folder are complete and consistent. If the rerun is merged later, update report sections 7.5 and the "500 images" wording.

## NOT to include
`unrelated/` (simulated data), `results/` raw maps (19 GB), `results.zip`.

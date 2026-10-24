# M4 XAI Benchmark — Replication & Extension
**Course**: DA351 – Explainable AI (XAI), IIIT Dharwad  
**Paper**: [M⁴: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods Across Metrics, Modalities and Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/m4-xai-paper.pdf) (NeurIPS 2023)

---

## Project Overview

This project replicates and extends the **M4 XAI Benchmark** from NeurIPS 2023. The paper evaluates the *faithfulness* of feature attribution (XAI) methods across multiple metrics, modalities (image and text), and model architectures.

**Random Seed**: All experiments use `seed=42` for reproducibility.

---

## Full M4 grid (all models in the paper)

`m4bench/` is a PyTorch port of the authors' pipeline
([M4_XAI_Benchmark](https://github.com/holyseven/M4_XAI_Benchmark) + [InterpretDL](https://github.com/PaddlePaddle/InterpretDL)),
covering every model, method and metric in the paper:

| Modality | Models | Methods | Metrics |
|---|---|---|---|
| Images (5000 ImageNet-val) | MobileNet-V3, VGG16, ResNet-50/101/152 | GradCAM, IG, SG | MoRF, ABPC, PScore, INFD, SynScore |
| | ViT-S/B/L, MAE-ViT-B | IG, SG, GA, BT-head, BT-token | MoRF, ABPC, PScore, INFD, SynScore |
| Text (MovieReview val) | DistilBERT, BERT-B/L, ERNIE-2.0, RoBERTa-B | IG, SG, LIME, GA, BT-head, BT-token | MoRF, ABPC, PScore, INFD |

```bash
JOBS=4 bash scripts/download_imagenet_train_10k.sh   # SynScore training set (Kaggle rate-limits: slow, resumable)
.venv/bin/python tests/smoke_test.py            # self-checks, no data needed
.venv/bin/python scripts/preflight.py            # data complete + all weights fetched (run_all.sh does this first)
bash run_all.sh                                  # everything (resumable)
LIMIT=500 bash run_all.sh                        # first 500 images only
SKIP_SYNSCORE=1 bash run_all.sh                  # without SynScore (needs no ImageNet-train images)
.venv/bin/python aggregate_results.py            # tables + Fig. 2-5 -> results/m4/summary/
```

Step by step: `train_text.py` → `run_text.py` → `run_vision.py` → `compute_pscore.py` →
`run_synscore.py train|eval` → `aggregate_results.py`.

**Settings taken from the authors' scripts** (they are not all in the paper):
100 perturbation percentiles for images (`run_eval.sh`), 101 for text (`percentile=1`);
IG 50 steps × 10 N(0,1) random baselines (images) / zero-embedding baseline (text);
SG 100 samples, noise 0.1·(max−min) (images) / 50 samples, 0.1·mean(embedding) (text);
LIME 1000 samples; BT 20 integration steps; INFD with 32/64/128-px patches at stride 8 (images) or
1–5-token windows (text); PScore = cosine to the cross-model mean; SynScore = AP of the explanation
against a pasted 60×60 yellow square; text models fine-tuned 10 epochs, batch 32, lr 3e-5, max length
512 (BERT-L 300).

**Matching the authors' setup.** The vision models are PyTorch mirrors of the authors' own
architectures (`m4bench/paddle_models.py`), loaded with the same Paddle weights files
(paddle-hapi / PaddleClas `.pdparams`, converted on load). This includes the authors' 8-block ViT-S.
MAE-ViT-B uses the official MAE fine-tuned checkpoint, which the authors converted to Paddle.
All models get InterpretDL's input pipeline, including ImageNet mean/std for the ViTs. The text
models start from the same pretrained checkpoints as PaddleNLP's. RoBERTa uses PaddleNLP's
pooler → dropout → linear head.

**What the paper and code leave open** (state these in the report):
- The authors swept GA/BT start layers (text: every model; vision: ViT-B and MAE) without saying
  which variant their figures show. Every variant is run and listed. `aggregate_results.py
  --layer_rule` picks the one used in Fig. 3/5; the default is highest ABPC.
- The authors published no PScore code for text. The vision procedure is applied to the four
  models that share the BERT vocabulary.
- Fine-tuning (text) and from-scratch training (SynScore) are stochastic. They follow the authors'
  recipes, but the trained weights will not be bit-identical to theirs.
- SynScore needs the 10k ImageNet-train images in `benchmark_data/imagenet_train_10k.txt`, placed at
  `data/imagenet_train_10k/<synset>/`.

**Compute:** expect days rather than hours on one 8 GB GPU for all 5000 images. Per image, IG costs
500 forward+backward passes and INFD 1235 forward passes, and ViT-L is the slowest model.
`--limit`, `--amp` and `--no_infd` trade fidelity for time. Every run resumes where it stopped.

## Phase 1: Early replication scripts (superseded by the full grid above)

These first-pass scripts use reduced settings (e.g. `--limit 500`, a single BERT model, a few methods).
They are **not** the paper reproduction; `run_all.sh` + `aggregate_results.py` are. Their outputs are not
committed: re-run them if you need them (`results/table1_faithful/` is created by `table1_resnet50.py`).

### Experiments Completed

| Experiment | Script | Output | Status |
|---|---|---|---|
| BERT-base MoRF (MovieReview) | `evaluate_explanations.py` | `morf_curve.png` | ✅ |
| VGG16 MoRF (ImageNet) | `evaluate_vgg16.py` | `morf_curve_vgg16.png` | ✅ |
| ResNet-50 Table 1 (reduced) | `table1_resnet50.py` | `results/table1_faithful/` | re-run; full Table 1 comes from `aggregate_results.py` → `results/m4/summary/table1_resnet50.md` |

### Key Results (Phase 1)

**BERT-base (NLP/Text)**:
- Integrated Gradients MoRF: Probability drops from `0.607` (0% masked) → `~0.495` (50% masked)
- Confirms steep initial decline described in the M4 paper

**VGG16 (Computer Vision)**:
- IG MoRF: `0.779` → `0.290` at 10% patches masked (−49% drop)
- Confirms that top-attributed patches are highly predictive

---

## Phase 2: Research Gap & Extension 🔨

### Research Gap: Masking Baseline Sensitivity

> The M4 paper uses a fixed grey-pixel `(127,127,127)` baseline for vision and `[PAD]` token for NLP. **No analysis was done on how sensitive faithfulness scores (MoRF, ABPC) are to this baseline choice.**

See [`research_gap.md`](research_gap.md) for the full 1-page explanation.

### Extension Experiments

| Experiment | Script | Output | Status |
|---|---|---|---|
| Vision baseline sensitivity (ResNet-50) | `phase2_vision_baseline.py` | `results/phase2/vision_*.png/csv` | 🔨 |
| NLP baseline sensitivity (BERT-base) | `phase2_nlp_baseline.py` | `results/phase2/nlp_*.png/csv` | 🔨 |

---

## Environment Setup

```bash
# Using uv (recommended)
uv pip install torch torchvision transformers captum datasets \
               scikit-learn matplotlib pillow pytorch-grad-cam opencv-python
```

Or with pip:
```bash
pip install torch torchvision transformers captum datasets \
            scikit-learn matplotlib pillow pytorch-grad-cam opencv-python
```

---

## Dataset

- **ImageNet M4_5000**: `data/M4_5000/` — 1000 synset subdirectories (ImageFolder format)
- **MovieReview**: `data/movies/` — ERASER benchmark format (`test.jsonl` + `docs/`)

---

## Running the Experiments

### Phase 1

```bash
# BERT-base training (if not already trained)
python train_bert.py

# BERT-base MoRF evaluation
python evaluate_explanations.py

# VGG16 MoRF evaluation
python evaluate_vgg16.py

# ResNet-50 Table 1 replication
python table1_resnet50.py --data_dir data/M4_5000 --limit 500
```

### Phase 2

```bash
# Vision baseline sensitivity (100 images, ~30 min on RTX 3050)
python phase2_vision_baseline.py --data_dir data/M4_5000 --n_images 100

# NLP baseline sensitivity (50 samples, ~20 min on RTX 3050)
python phase2_nlp_baseline.py --data_dir data/movies --model_dir models/bert-base-movies
```

---

## Repository Structure

```
m4_xai_benchmark/
├── data/
│   ├── M4_5000/          # ImageNet M4 5000-image subset
│   └── movies/           # MovieReview ERASER dataset
├── models/
│   └── bert-base-movies/ # Fine-tuned BERT checkpoint
├── results/
│   ├── table1_faithful/  # Phase 1 ResNet-50 Table 1 results
│   └── phase2/           # Phase 2 baseline sensitivity results
├── train_bert.py         # BERT-base fine-tuning on MovieReview
├── evaluate_explanations.py  # BERT MoRF evaluation (Phase 1)
├── evaluate_vgg16.py     # VGG16 MoRF evaluation (Phase 1)
├── table1_resnet50.py    # ResNet-50 Table 1 replication (Phase 1)
├── benchmark_faithfulness.py  # Generic MoRF+ABPC benchmark
├── phase2_vision_baseline.py  # Phase 2: Vision baseline sensitivity
├── phase2_nlp_baseline.py     # Phase 2: NLP baseline sensitivity
├── research_gap.md       # 1-page research gap explanation (Phase 2)
├── AI_USAGE.md           # AI tool usage log
└── README.md
```

---

## Reproducibility

- **Random seed**: `42` (all scripts)
- **Dataset link**: M4_5000 (from [M4 benchmark repo](https://github.com/PaddlePaddle/InterpretDL)), MovieReview (ERASER)
- **Requirements**: See environment setup above
- **Instructions**: See "Running the Experiments" section above
# M4-XAI

# M4 XAI Benchmark — Replication & Extension
**Course**: DA351 – Explainable AI (XAI), IIIT Dharwad  
**Paper**: [M⁴: A Unified XAI Benchmark for Faithfulness Evaluation of Feature Attribution Methods Across Metrics, Modalities and Models](https://proceedings.neurips.cc/paper_files/paper/2023/file/m4-xai-paper.pdf) (NeurIPS 2023)

---

## Project Overview

This project replicates and extends the **M4 XAI Benchmark** from NeurIPS 2023. The paper evaluates the *faithfulness* of feature attribution (XAI) methods across multiple metrics, modalities (image and text), and model architectures.

**Random Seed**: All experiments use `seed=42` for reproducibility.

---

## Phase 1: Replication ✅

### Experiments Completed

| Experiment | Script | Output | Status |
|---|---|---|---|
| BERT-base MoRF (MovieReview) | `evaluate_explanations.py` | `morf_curve.png` | ✅ |
| VGG16 MoRF (ImageNet) | `evaluate_vgg16.py` | `morf_curve_vgg16.png` | ✅ |
| ResNet-50 Table 1 (M4_5000) | `table1_resnet50.py` | `results/table1_faithful/` | ✅ |

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

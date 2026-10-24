#!/usr/bin/env bash
# Full M4 grid: 9 vision models + 5 text models, all methods and metrics of the paper.
# Every stage is resumable: re-running skips finished samples.
#
#   bash run_all.sh                 # everything
#   LIMIT=500 bash run_all.sh       # first 500 images / all 200 val docs, for a faster pass
#
# Needs: data/M4_5000/<synset>/*.JPEG (benchmark_data/imagenet_val_5k.txt), data/movies,
#        and for SynScore data/imagenet_train_10k/<synset>/*.JPEG (benchmark_data/imagenet_train_10k.txt).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=.venv/bin/python
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
mkdir -p results/m4/logs

# ── Preflight: data complete + all weights fetched, before any long job starts ───
# SynScore needs all 10,000 ImageNet-train images; a partial set would give wrong SynScores, so it is an
# error unless you opt out explicitly with SKIP_SYNSCORE=1 (then SynScore/AvgScore are incomplete).
if [[ ${SKIP_SYNSCORE:-0} == 1 ]]; then
    $PY scripts/preflight.py --no_synscore
else
    $PY scripts/preflight.py
fi

# ── Text: fine-tune, explain + evaluate ─────────────────────────────────────
$PY train_text.py --models all                                   2>&1 | tee -a results/m4/logs/train_text.log
$PY run_text.py   --models all                                   2>&1 | tee -a results/m4/logs/run_text.log
$PY compute_pscore.py --modality text                            2>&1 | tee -a results/m4/logs/pscore_text.log

# ── Vision: explain + evaluate (ResNet-50 also gets the Table 1 baselines) ───
$PY run_vision.py --models resnet50 --baselines $LIMIT_ARG       2>&1 | tee -a results/m4/logs/run_vision.log
$PY run_vision.py --models mobilenet vgg16 resnet101 resnet152 vit_small vit_base mae_vit_base $LIMIT_ARG \
                                                                 2>&1 | tee -a results/m4/logs/run_vision.log
$PY run_vision.py --models vit_large --grad_bs 5 --bs 32 $LIMIT_ARG 2>&1 | tee -a results/m4/logs/run_vision.log
$PY compute_pscore.py --modality vision                          2>&1 | tee -a results/m4/logs/pscore_vision.log

# ── SynScore: needs the 10k ImageNet-train images ───────────────────────────
if [[ ${SKIP_SYNSCORE:-0} != 1 ]]; then
    $PY run_synscore.py train --models mobilenet resnet50 resnet101 resnet152 2>&1 | tee -a results/m4/logs/synscore.log
    $PY run_synscore.py train --models vgg16 vit_small vit_base vit_large mae_vit_base --micro_bs 16 2>&1 | tee -a results/m4/logs/synscore.log
    $PY run_synscore.py eval  --models all                       2>&1 | tee -a results/m4/logs/synscore.log
else
    echo "[skip] SynScore (SKIP_SYNSCORE=1)"
fi

$PY aggregate_results.py

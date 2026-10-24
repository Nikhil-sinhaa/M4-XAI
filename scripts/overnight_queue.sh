#!/usr/bin/env bash
# Everything that does not need the SynScore training images, one GPU job at a time, cheapest first.
# First pass: LIMIT images per vision model (default 500) and TEXT_LIMIT reviews per text model (default 100).
# All stores are sized for the full set and every step resumes, so re-running with LIMIT=5000 TEXT_LIMIT=200
# (or unset) extends the same results instead of starting over.
#   nohup bash scripts/overnight_queue.sh > results/m4/logs/queue.log 2>&1 &
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
LIMIT=${LIMIT:-500}
TEXT_LIMIT=${TEXT_LIMIT:-100}
mkdir -p results/m4/logs
step() {
    local name=$1 rc; shift
    echo "=== $(date '+%F %T') START $name"
    "$@" >> "results/m4/logs/$name.log" 2>&1
    rc=$?
    echo "=== $(date '+%F %T') END   $name (exit $rc)$([[ $rc != 0 ]] && echo '  <-- FAILED')"
}
vstep() { step "v_$1" $PY run_vision.py --models "$@" --limit "$LIMIT"; }
tstep() { step "t_train_$1" $PY train_text.py --models "$1"; step "t_run_$1" $PY run_text.py --models "$1" --limit "$TEXT_LIMIT"; }

vstep resnet50 --baselines            # Table 1
tstep distilbert
tstep bert_base
vstep vit_small
vstep vgg16
vstep resnet101
vstep resnet152
vstep vit_base
vstep mae_vit_base
tstep ernie_2_base
tstep roberta_base
tstep bert_large
step pscore_vision $PY compute_pscore.py --modality vision
step pscore_text   $PY compute_pscore.py --modality text
step aggregate     $PY aggregate_results.py
vstep vit_large --grad_bs 5 --bs 32   # slowest model last
step pscore_vision2 $PY compute_pscore.py --modality vision
step aggregate2     $PY aggregate_results.py
echo "=== $(date '+%F %T') QUEUE DONE"

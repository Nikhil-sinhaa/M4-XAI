# Sourced by every job. Compute nodes may be offline, so Hugging Face is used from the local cache.
cd "$HOME/m4_xai"
export HF_HOME="$HOME/m4_xai/.hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
PY=.venv/bin/python
LIMIT=${LIMIT:-500}
TEXT_LIMIT=${TEXT_LIMIT:-100}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
declare -A WFILE=([resnet50]=resnet50.pdparams [mobilenet]=mobilenet_v3_large_x1.0.pdparams
  [resnet101]=resnet101.pdparams [resnet152]=resnet152.pdparams [vgg16]=vgg16.pdparams
  [vit_small]=ViT_small_patch16_224_pretrained.pdparams [vit_base]=ViT_base_patch16_224_pretrained.pdparams
  [vit_large]=ViT_large_patch16_224_pretrained.pdparams [mae_vit_base]=mae_finetuned_vit_base.pth)
# weights are fetched on the login node by scripts/fetch_weights.sh; wait for the finished file
waitw() { until [[ -s models/paddle_weights/${WFILE[$1]} ]]; do sleep 60; done; }
vrun() { waitw "$1"; echo "=== $(date '+%F %T') START vision $*"; $PY run_vision.py --models "$@" --limit "$LIMIT"; echo "=== $(date '+%F %T') END vision $1 (exit $?)"; }
trun() { echo "=== $(date '+%F %T') START text $1"
         $PY train_text.py --models "$1" && $PY run_text.py --models "$1" --limit "$TEXT_LIMIT"
         echo "=== $(date '+%F %T') END text $1 (exit $?)"; }

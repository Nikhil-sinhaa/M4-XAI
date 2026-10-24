#!/usr/bin/env bash
# Resumable, parallel download of the pretrained vision weights into models/paddle_weights/ (the same
# files m4bench/paddle_models.py would fetch, but torch.hub's downloader cannot resume and the Baidu
# mirrors are slow). Files are written as <name>.part and renamed only when complete, so a half file
# is never mistaken for a finished one. Re-run to resume.
#
#   bash scripts/fetch_weights.sh            # all
#   bash scripts/fetch_weights.sh resnet50   # one (name as in PADDLE_URLS, or "mae")
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT=models/paddle_weights
mkdir -p "$OUT"
declare -A URL=(
  [resnet50]=https://paddle-hapi.bj.bcebos.com/models/resnet50.pdparams
  [mobilenet]=https://paddle-hapi.bj.bcebos.com/models/mobilenet_v3_large_x1.0.pdparams
  [resnet101]=https://paddle-hapi.bj.bcebos.com/models/resnet101.pdparams
  [vit_small]=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_small_patch16_224_pretrained.pdparams
  [resnet152]=https://paddle-hapi.bj.bcebos.com/models/resnet152.pdparams
  [vgg16]=https://paddle-hapi.bj.bcebos.com/models/vgg16.pdparams
  [vit_base]=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_base_patch16_224_pretrained.pdparams
  [mae]=https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth
  [vit_large]=https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ViT_large_patch16_224_pretrained.pdparams
)
ORDER=(resnet50 mobilenet resnet101 vit_small resnet152 vgg16 vit_base mae vit_large)
[[ $# -gt 0 ]] && ORDER=("$@")

get() {
    local url=$1 dest="$OUT/${1##*/}"
    [[ -s $dest ]] && return 0
    local ok=0
    for attempt in $(seq 1 200); do
        if curl -sS -L -C - --connect-timeout 20 --speed-time 60 --speed-limit 1000 -o "$dest.part" "$url"; then
            ok=1; break
        fi
        sleep 5
    done
    if (( ok )) && [[ -s $dest.part ]]; then mv "$dest.part" "$dest" && echo "done ${dest##*/}"
    else echo "INCOMPLETE ${dest##*/}" >&2; fi
}
for n in "${ORDER[@]}"; do
    get "${URL[$n]}" &
done
wait
ls -la "$OUT"

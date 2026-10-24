#!/usr/bin/env bash
# Download exactly the 10,000 ImageNet-train images listed in benchmark_data/imagenet_train_10k.txt
# (the M4 SynScore training set) from Kaggle's imagenet-object-localization-challenge, one file at a time.
# Requires ~/.kaggle/access_token and the competition rules accepted.
#
# Kaggle rate-limits this endpoint (HTTP 429), so the defaults are gentle: few parallel jobs, exponential
# back-off that honours Retry-After, and repeated passes over whatever is still missing. Existing valid
# JPEGs are skipped, so re-running resumes. Exits non-zero while any image is missing.
#
#   JOBS=4 PASSES=10 bash scripts/download_imagenet_train_10k.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
LIST=benchmark_data/imagenet_train_10k.txt
OUT=data/imagenet_train_10k
JOBS=${JOBS:-4}
PASSES=${PASSES:-10}
[[ -s ~/.kaggle/access_token ]] || { echo "missing ~/.kaggle/access_token" >&2; exit 2; }
# Pass the token to curl through a private header file so it never appears in the process list.
export HDR; HDR=$(mktemp); chmod 600 "$HDR"; trap 'rm -f "$HDR"' EXIT
printf 'Authorization: Bearer %s\n' "$(cat ~/.kaggle/access_token)" > "$HDR"
export OUT

is_jpeg() { [[ -s $1 ]] && head -c 3 "$1" | od -An -tx1 | grep -q "ff d8 ff"; }
export -f is_jpeg

fetch() {
    local rel=$1 dest="$OUT/$1"
    is_jpeg "$dest" && return 0
    mkdir -p "$(dirname "$dest")"
    local url="https://www.kaggle.com/api/v1/competitions/data/download/imagenet-object-localization-challenge/ILSVRC%2FData%2FCLS-LOC%2Ftrain%2F${rel//\//%2F}"
    local attempt wait hdrs code
    hdrs=$(mktemp)
    for attempt in 1 2 3 4 5 6; do
        code=$(curl -sS -L -H @"$HDR" -D "$hdrs" -o "$dest.part" -w '%{http_code}' "$url" 2>/dev/null) || code=000
        if [[ $code == 200 ]] && is_jpeg "$dest.part"; then
            mv "$dest.part" "$dest"; rm -f "$hdrs"; return 0
        fi
        wait=$(( 2 ** attempt ))
        if [[ $code == 429 ]]; then
            ra=$(grep -i '^retry-after:' "$hdrs" | tail -1 | tr -dc '0-9')
            [[ -n ${ra:-} ]] && wait=$(( ra > wait ? ra : wait ))
            wait=$(( wait + RANDOM % 5 ))
        fi
        sleep "$wait"
    done
    rm -f "$dest.part" "$hdrs"
    echo "FAILED $rel" >&2
    return 1
}
export -f fetch

missing_list() {
    while read -r rel; do is_jpeg "$OUT/$rel" || echo "$rel"; done < "$LIST"
}

for pass in $(seq 1 "$PASSES"); do
    todo=$(mktemp); missing_list > "$todo"
    n=$(wc -l < "$todo")
    echo "pass $pass/$PASSES: $n images missing"
    if (( n == 0 )); then rm -f "$todo"; break; fi
    xargs -a "$todo" -P "$JOBS" -I{} bash -c 'fetch "$@"' _ {} || true
    rm -f "$todo"
    sleep 30       # let the rate limit cool down between passes
done

have=$(missing_list | wc -l)
total=$(wc -l < "$LIST")
echo "done: $(( total - have )) / $total images in $OUT"
(( have == 0 ))
